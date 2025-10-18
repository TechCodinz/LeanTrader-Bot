"""
Core trading modules

This package keeps its top-level import lightweight to avoid importing heavy
submodules during package initialization. Import subpackages/modules directly,
e.g. `from core.features.ultra_advanced import compute_ultra_advanced`.
"""

# Expose subpackages intentionally; avoid eager submodule imports here.
__all__ = [
    "features",
]
