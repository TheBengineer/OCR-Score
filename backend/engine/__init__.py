"""OCR engine plugin interface and registry.

All OCR engines must subclass ``OCREngine`` and implement its abstract methods.
The ``EngineRegistry`` singleton discovers and manages available engine plugins.

Typical usage::

    from backend.engine import OCREngine, EngineRegistry, registry

    class MyEngine(OCREngine):
        engine_id = "my-engine"
        display_name = "My Custom OCR"
        version = "1.0.0"
        ...

    registry.register(MyEngine)
"""

from backend.engine.base import OCREngine
from backend.engine.registry import EngineRegistry, EngineRegistryError, registry

__all__ = [
    "OCREngine",
    "EngineRegistry",
    "EngineRegistryError",
    "registry",
]
