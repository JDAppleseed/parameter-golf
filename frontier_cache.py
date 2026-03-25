from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_HASH_PRIMES = np.array(
    [np.uint64(36313), np.uint64(27191), np.uint64(51647), np.uint64(81929), np.uint64(131071)],
    dtype=np.uint64,
)


@dataclass(frozen=True)
class CausalCacheConfig:
    mode: str
    max_order: int
    alpha: float
    min_count: int
    buckets: int
    mixing: str = "fixed"
    count_smoothing: float = 4.0

    def validate(self) -> None:
        if self.mode not in {"off", "ngram7", "ppm"}:
            raise ValueError(f"CAUSAL_CACHE_MODE must be one of off/ngram7/ppm, got {self.mode!r}")
        if self.max_order < 2:
            raise ValueError(f"CAUSAL_CACHE_MAX_ORDER must be >= 2, got {self.max_order}")
        if self.alpha < 0.0 or self.alpha > 1.0:
            raise ValueError(f"CAUSAL_CACHE_ALPHA must be in [0, 1], got {self.alpha}")
        if self.min_count < 1:
            raise ValueError(f"CAUSAL_CACHE_MIN_COUNT must be >= 1, got {self.min_count}")
        if self.buckets <= 0 or self.buckets & (self.buckets - 1):
            raise ValueError(f"CAUSAL_CACHE_BUCKETS must be a positive power of two, got {self.buckets}")
        if self.mixing not in {"fixed", "count"}:
            raise ValueError(f"CAUSAL_CACHE_MIXING must be fixed or count, got {self.mixing!r}")
        if self.count_smoothing <= 0.0:
            raise ValueError(f"CAUSAL_CACHE_COUNT_SMOOTHING must be > 0, got {self.count_smoothing}")

    @property
    def orders(self) -> list[int]:
        if self.mode == "off":
            return []
        if self.mode == "ngram7":
            return [self.max_order]
        return list(range(2, self.max_order + 1))


class ScoreFirstCausalCache:
    """Deterministic backward-looking cache with an explicit score-then-commit API.

    Call `reset()` whenever the evaluator wants to enforce a document boundary.
    """

    def __init__(self, config: CausalCacheConfig):
        config.validate()
        self.config = config
        self.mask = np.uint64(config.buckets - 1)
        self.ctx_tables = {order: np.zeros((config.buckets,), dtype=np.uint32) for order in config.orders}
        self.full_tables = {order: np.zeros((config.buckets,), dtype=np.uint32) for order in config.orders}
        self._pending_positions: np.ndarray | None = None

    def reset(self) -> None:
        for table in self.ctx_tables.values():
            table.fill(0)
        for table in self.full_tables.values():
            table.fill(0)
        self._pending_positions = None

    def score_segment(
        self,
        token_stream: np.ndarray,
        global_target_positions: np.ndarray,
        model_target_probs: np.ndarray,
    ) -> np.ndarray:
        if self.config.mode == "off":
            return model_target_probs
        if self._pending_positions is not None:
            raise RuntimeError("cache score/commit ordering violated: commit the pending segment before scoring again")

        mixed = np.array(model_target_probs, copy=True, dtype=np.float64)
        best_ng = np.zeros_like(mixed)
        best_ctx_count = np.zeros_like(mixed)
        matched = np.zeros_like(mixed, dtype=bool)

        for order in reversed(self.config.orders):
            ctx_width = order - 1
            valid = (global_target_positions >= ctx_width) & ~matched
            if not valid.any():
                continue
            idx = np.nonzero(valid)[0]
            positions = global_target_positions[idx]
            ctx_hash = np.zeros(len(positions), dtype=np.uint64)
            for offset in range(ctx_width):
                tok = token_stream[positions - (ctx_width - offset)].astype(np.uint64)
                ctx_hash ^= tok * _HASH_PRIMES[offset % len(_HASH_PRIMES)]
            ctx_key = (ctx_hash & self.mask).astype(np.int64)
            tgt = token_stream[positions].astype(np.uint64)
            full_key = ((ctx_hash ^ (tgt * _HASH_PRIMES[ctx_width % len(_HASH_PRIMES)])) & self.mask).astype(np.int64)

            ctx_counts = self.ctx_tables[order][ctx_key].astype(np.float64)
            full_counts = self.full_tables[order][full_key].astype(np.float64)
            can_mix = ctx_counts >= float(self.config.min_count)
            if not can_mix.any():
                continue
            chosen = idx[can_mix]
            p_ng = np.minimum(full_counts[can_mix], ctx_counts[can_mix]) / np.maximum(ctx_counts[can_mix], 1.0)
            best_ng[chosen] = np.clip(p_ng, 0.0, 1.0)
            best_ctx_count[chosen] = ctx_counts[can_mix]
            matched[chosen] = True

        mix_idx = np.nonzero(matched)[0]
        if mix_idx.size:
            if self.config.mixing == "count":
                alpha_vec = self.config.alpha * (
                    best_ctx_count[mix_idx] / (best_ctx_count[mix_idx] + self.config.count_smoothing)
                )
            else:
                alpha_vec = np.full(mix_idx.size, self.config.alpha, dtype=np.float64)
            mixed[mix_idx] = (1.0 - alpha_vec) * mixed[mix_idx] + alpha_vec * best_ng[mix_idx]

        self._pending_positions = np.array(global_target_positions, copy=True, dtype=np.int64)
        return mixed

    def commit_segment(self, token_stream: np.ndarray, global_target_positions: np.ndarray) -> None:
        if self.config.mode == "off":
            return
        if self._pending_positions is None:
            raise RuntimeError("cache score/commit ordering violated: cannot commit before score_segment")
        if not np.array_equal(self._pending_positions, np.asarray(global_target_positions, dtype=np.int64)):
            raise RuntimeError("cache score/commit ordering violated: commit_segment received a different segment")

        for order in self.config.orders:
            ctx_width = order - 1
            valid = global_target_positions >= ctx_width
            if not valid.any():
                continue
            positions = global_target_positions[valid]
            ctx_hash = np.zeros(len(positions), dtype=np.uint64)
            for offset in range(ctx_width):
                tok = token_stream[positions - (ctx_width - offset)].astype(np.uint64)
                ctx_hash ^= tok * _HASH_PRIMES[offset % len(_HASH_PRIMES)]
            ctx_key = (ctx_hash & self.mask).astype(np.int64)
            tgt = token_stream[positions].astype(np.uint64)
            full_key = ((ctx_hash ^ (tgt * _HASH_PRIMES[ctx_width % len(_HASH_PRIMES)])) & self.mask).astype(np.int64)
            np.add.at(self.ctx_tables[order], ctx_key, 1)
            np.add.at(self.full_tables[order], full_key, 1)

        self._pending_positions = None


def causal_cache_from_env(env: dict[str, str] | None = None) -> ScoreFirstCausalCache | None:
    source = env if env is not None else {}
    mode = source.get("CAUSAL_CACHE_MODE", "off")
    config = CausalCacheConfig(
        mode=mode,
        max_order=int(source.get("CAUSAL_CACHE_MAX_ORDER", "7")),
        alpha=float(source.get("CAUSAL_CACHE_ALPHA", "0.40")),
        min_count=int(source.get("CAUSAL_CACHE_MIN_COUNT", "2")),
        buckets=int(source.get("CAUSAL_CACHE_BUCKETS", "4194304")),
        mixing=source.get("CAUSAL_CACHE_MIXING", "fixed"),
        count_smoothing=float(source.get("CAUSAL_CACHE_COUNT_SMOOTHING", "4.0")),
    )
    if config.mode == "off":
        return None
    return ScoreFirstCausalCache(config)
