"""Explicit per-call backend selection; a missing extension has a Python fallback."""
from importlib import import_module


def resolve_backend(backend):
    if backend not in ('auto', 'python', 'rust'):
        raise ValueError("backend must be 'auto', 'python', or 'rust'.")
    if backend == 'python':
        return None
    try:
        return import_module('ov_radiomics._native')
    except ModuleNotFoundError as exc:
        # Only an absent extension permits fallback. A broken DLL/dependency must surface.
        if exc.name != 'ov_radiomics._native':
            raise
        if backend == 'rust':
            raise ImportError('Rust extension is not built. Install a compiled wheel or run maturin develop --release.') from exc
        return None


def backend_info():
    """Report the actual default metric backend; does not change process-wide settings."""
    native = resolve_backend('auto')
    return {'metrics_backend': 'rust' if native is not None else 'python',
            'rust_core_version': native.CORE_VERSION if native is not None else None}
