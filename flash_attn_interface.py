"""Repo-local compatibility shim for FlashAttention imports.

The frontier trainers import `flash_attn_interface.flash_attn_func`. Depending on
the installed flash-attn package version, that symbol may live at either:

- `flash_attn.flash_attn_interface.flash_attn_func`
- `flash_attn.flash_attn_func`

Keeping this shim in-repo makes the cloud setup less brittle without changing the
actual attention kernel used at runtime.
"""

from __future__ import annotations

try:
    from flash_attn.flash_attn_interface import flash_attn_func
except ImportError:  # pragma: no cover - depends on installed flash-attn version
    from flash_attn import flash_attn_func

__all__ = ["flash_attn_func"]
