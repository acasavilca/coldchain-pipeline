"""
Central numba JIT configuration.

Every compiled function in the simulator is decorated with ``njit`` from this
module instead of ``numba.njit``, so caching and fast-math are set in one place.

    @njit                      # cache + default fast-math
    @njit(fastmath=False)      # opt a single function out of fast-math

Environment (set by run.py's --no-cache / --clear-cache / --fastmath flags):

    COLDCHAIN_NUMBA_CACHE      1 (default) | 0   -- on-disk compilation cache
    COLDCHAIN_NUMBA_FASTMATH   safe (default) | full | off

Cache invalidation: numba keys its cache on each source file's mtime only, so
editing constants.py or matrix_index_maps.py (imported global ints) would leave
stale compiled callers in other modules. To avoid that, the cache directory is
named after a hash of every *.py / *.json under the simulation folder: any edit
selects a fresh directory (and older ones are pruned), so no manual cache
clearing is ever required. ``--no-cache`` skips the cache entirely.

Fast-math levels:
    safe  reassoc, contract, arcp, nsz, afn   (keeps NaN/Inf semantics)
    full  fastmath=True (adds nnan/ninf; NaN/Inf checks may be optimised away)
    off   no fast-math
"""
import hashlib
import os
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE_ROOT = HERE / ".numba_cache"

_SAFE_FLAGS = {"reassoc", "contract", "arcp", "nsz", "afn"}


def source_fingerprint() -> str:
    h = hashlib.sha256()
    for f in sorted(list(HERE.glob("*.py")) + list(HERE.glob("configs/*.json"))):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def clear_cache() -> int:
    """Delete every on-disk numba cache under the simulation folder. Returns dirs removed."""
    n = 0
    if CACHE_ROOT.exists():
        shutil.rmtree(CACHE_ROOT, ignore_errors=True)
        n += 1
    for d in HERE.rglob("__pycache__"):
        for f in list(d.glob("*.nbi")) + list(d.glob("*.nbc")):
            f.unlink(missing_ok=True)
            n += 1
    return n


CACHE_ENABLED = os.environ.get("COLDCHAIN_NUMBA_CACHE", "1") != "0"
FASTMATH_LEVEL = os.environ.get("COLDCHAIN_NUMBA_FASTMATH", "safe").lower()

if CACHE_ENABLED:
    # numba's own cache key ignores both fast-math flags and imported globals, so the
    # folder is named <source hash>-<fastmath level>; folders of other source hashes are stale.
    _src_hash = source_fingerprint()
    _dir = CACHE_ROOT / f"{_src_hash}-{FASTMATH_LEVEL}"
    _dir.mkdir(parents=True, exist_ok=True)
    for old in CACHE_ROOT.iterdir():
        if old.is_dir() and not old.name.startswith(_src_hash):
            shutil.rmtree(old, ignore_errors=True)
    os.environ["NUMBA_CACHE_DIR"] = str(_dir)

import numba  # noqa: E402  (after NUMBA_CACHE_DIR is set)

if CACHE_ENABLED:
    numba.config.CACHE_DIR = str(_dir)

if FASTMATH_LEVEL == "full":
    _FASTMATH = True
elif FASTMATH_LEVEL == "off":
    _FASTMATH = False
else:
    _FASTMATH = set(_SAFE_FLAGS)


def njit(*args, **kwargs):
    opts = {"cache": CACHE_ENABLED, "fastmath": _FASTMATH}
    opts.update(kwargs)
    if args and callable(args[0]):
        return numba.njit(**opts)(args[0])
    return lambda f: numba.njit(*args, **opts)(f)
