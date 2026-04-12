#!/usr/bin/env bash

log() {
  printf '[runpod_frontier] %s\n' "$*"
}

die() {
  printf '[runpod_frontier] ERROR: %s\n' "$*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

abs_path() {
  python3 - "$1" <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).expanduser().resolve())
PY
}

slugify_name() {
  python3 - "$1" <<'PY'
import re
import sys

value = sys.argv[1].strip()
slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
print(slug or "run")
PY
}

python_import_available() {
  python3 -c "import $1" >/dev/null 2>&1
}

free_disk_bytes() {
  python3 - "$1" <<'PY'
from pathlib import Path
import shutil
import sys

path = Path(sys.argv[1]).expanduser().resolve()
target = path if path.exists() else path.parent
print(shutil.disk_usage(target).free)
PY
}

format_bytes_gib() {
  python3 - "$1" <<'PY'
import sys

value = int(sys.argv[1])
print(f"{value / (1024 ** 3):.1f} GiB")
PY
}
