"""Engine registry — in-memory singleton that discovers and manages OCR engine
plugins.

Usage
-----
    from backend.engine import registry

    # Discover all available engines (entry_points + backend/engine/ directory)
    registry.discover()

    # Get a specific engine by ID
    engine = registry.get("mock")

    # List all registered engines
    for eng in registry.list():
        print(eng.engine_id, eng.display_name, eng.version)

Engines can also be registered programmatically:

    from backend.mock_engine import MockEngine
    registry.register(MockEngine)
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from backend.engine.base import OCREngine


class EngineRegistryError(Exception):
    """Raised when an engine operation fails (unknown ID, duplicate, etc.)."""


class EngineRegistry:
    """Singleton in-memory registry of OCR engine classes.

    The registry holds engine *classes*, not instances.  ``get()`` creates a
    fresh ``OCREngine`` instance on each call.
    """

    _instance: EngineRegistry | None = None

    def __new__(cls) -> EngineRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._engines = {}
        return cls._instance

    def __init__(self) -> None:
        # __new__ handles singleton init; guard against re-init
        if not hasattr(self, "_engines"):
            self._engines: dict[str, type[OCREngine]] = {}

    # ── Public API ─────────────────────────────────────────────────────────

    def register(self, engine_cls: type[OCREngine]) -> None:
        """Register an OCR engine class.

        The class must have ``engine_id``, ``display_name``, and ``version``
        class attributes and be a concrete (non-abstract) ``OCREngine``
        subclass.

        Args:
            engine_cls: The engine class to register.

        Raises:
            EngineRegistryError: If the class is not an ``OCREngine`` subclass,
                is abstract, or if an engine with the same ``engine_id`` is
                already registered.
        """
        self._validate_engine_cls(engine_cls)

        engine_id = engine_cls.engine_id
        if engine_id in self._engines:
            msg = f"Engine '{engine_id}' is already registered"
            raise EngineRegistryError(msg)

        self._engines[engine_id] = engine_cls

    def get(self, engine_id: str) -> OCREngine:
        """Get a fresh instance of the engine with the given ID.

        Args:
            engine_id: The engine's unique identifier.

        Returns:
            A new instance of the registered engine class.

        Raises:
            EngineRegistryError: If no engine is registered with that ID.
        """
        cls = self._engines.get(engine_id)
        if cls is None:
            msg = f"No engine registered with ID '{engine_id}'"
            raise EngineRegistryError(msg)
        return cls()

    def list(self) -> list[OCREngine]:
        """Return a list of fresh instances for all registered engines.

        Returns:
            A list of ``OCREngine`` instances, one per registered class.
        """
        return [cls() for cls in self._engines.values()]

    def discover(self) -> None:
        """Scan for ``OCREngine`` subclasses in three locations:

        1. ``importlib.metadata.entry_points(group="ocrscore.engines")`` —
           discovers pip-installed third-party plugins.
        2. ``backend`` package — scans modules at the backend package level
           (e.g. ``backend.mock_engine``).
        3. ``backend.engine`` package — scans the local engine plugin directory.

        Already-registered engines are not duplicated.
        """
        # ── 1. Entry points (pip-installed plugins) ───────────────────────
        try:
            entry_points = importlib.metadata.entry_points(group="ocrscore.engines")
        except importlib.metadata.PackageNotFoundError:
            entry_points = []

        for ep in entry_points:
            try:
                cls = ep.load()
            except Exception:  # noqa: BLE001
                continue
            if self._is_concrete_engine(cls) and cls.engine_id not in self._engines:
                self._engines[cls.engine_id] = cls

        # ── 2. Local backend/ package scan ────────────────────────────────
        self._scan_package("backend", engines=self._engines)

        # ── 3. Local backend/engine/ package scan ─────────────────────────
        self._scan_package(
            "backend.engine",
            skip_modules=frozenset({
                "backend.engine.__init__",
                "backend.engine.base",
                "backend.engine.normalized_schema",
                "backend.engine.registry",
            }),
            engines=self._engines,
        )

    # ── Internals ──────────────────────────────────────────────────────────

    @staticmethod
    def _scan_package(
        package_name: str,
        *,
        skip_modules: frozenset[str] = frozenset(),
        engines: dict[str, type[OCREngine]],
    ) -> None:
        """Import every module in *package_name* and register any concrete
        ``OCREngine`` subclasses found.

        Args:
            package_name: Dotted package name to scan (e.g. ``"backend.engine"``).
            skip_modules: Set of fully-qualified module names to skip.
            engines: The registry's engine dict to populate.
        """
        try:
            pkg = importlib.import_module(package_name)
        except ImportError:
            return

        if not hasattr(pkg, "__path__"):
            return

        for _finder, module_name, _ispkg in pkgutil.iter_modules(
            pkg.__path__,
            prefix=f"{package_name}.",
        ):
            if module_name in skip_modules:
                continue

            try:
                mod = importlib.import_module(module_name)
            except Exception:  # noqa: BLE001
                continue

            for _name, obj in inspect.getmembers(mod, EngineRegistry._is_concrete_engine):
                engine_id = obj.engine_id  # type: ignore[attr-defined]
                if engine_id not in engines:
                    engines[engine_id] = obj  # type: ignore[arg-type]

    @staticmethod
    def _validate_engine_cls(engine_cls: type[OCREngine]) -> None:
        """Validate that *engine_cls* is a usable concrete engine class."""
        from backend.engine.base import OCREngine as OCREngineABC

        if not (isinstance(engine_cls, type) and issubclass(engine_cls, OCREngineABC)):
            msg = f"{engine_cls.__name__} is not a subclass of OCREngine"
            raise EngineRegistryError(msg)

        if engine_cls is OCREngineABC:
            msg = "Cannot register the abstract base class OCREngine itself"
            raise EngineRegistryError(msg)

        for attr in ("engine_id", "display_name", "version"):
            if not hasattr(engine_cls, attr):
                msg = (
                    f"{engine_cls.__name__} is missing required "
                    f"class attribute '{attr}'"
                )
                raise EngineRegistryError(msg)

        if inspect.isabstract(engine_cls):
            msg = f"{engine_cls.__name__} is abstract and cannot be registered"
            raise EngineRegistryError(msg)

    @staticmethod
    def _is_concrete_engine(obj: Any) -> bool:
        """Return True if *obj* is a concrete (non-abstract) ``OCREngine`` subclass."""
        from backend.engine.base import OCREngine as OCREngineABC

        return (
            isinstance(obj, type)
            and issubclass(obj, OCREngineABC)
            and obj is not OCREngineABC
            and not inspect.isabstract(obj)
        )


# Module-level singleton — all users share the same instance.
registry: EngineRegistry = EngineRegistry()
