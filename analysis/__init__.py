"""Analysis modules; truth-only tools load without reconstruction dependencies."""

from importlib import import_module

__all__ = ["metrics", "efficiency", "sensitivity"]


def __getattr__(name):
    if name in __all__:
        value = import_module("." + name, __name__)
        globals()[name] = value
        return value
    raise AttributeError(name)
