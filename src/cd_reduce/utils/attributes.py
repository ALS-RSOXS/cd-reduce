"""Companion decorator for the CdRsoxsLoader class."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _make_process_var_property(key: str) -> property:
    """Create a property getter/setter pair bound to `instance.process_vars[key]`."""

    def getter(instance: Any) -> Any:
        """Read value from ``instance.process_vars[key]``."""
        return instance.process_vars[key]

    def setter(instance: Any, value: Any) -> None:
        """Set value in ``instance.process_vars[key]`` and mark dirty."""
        instance.process_vars[key] = value
        if "reprocess_vars" not in key:
            instance.process_vars["reprocess_vars"] = True

    return property(getter, setter)


def process_vars_properties[T: type](cls: T) -> T:
    """Automatically create properties for each key in ``process_vars_defaults``.

    Follows the same ergonomic pattern as ``pxr_reduce.utils.attributes``:
    setting any process variable updates ``instance.process_vars[key]`` and
    marks ``instance.process_vars["reprocess_vars"] = True``.
    """
    defaults: Mapping[str, Any] | None = getattr(
        cls, "process_vars_defaults", None
    ) or getattr(cls, "process_vars", None)

    if isinstance(defaults, Mapping):
        for key in defaults:
            setattr(cls, key, _make_process_var_property(key))

    return cls
