"""Registry of public function names across all rolling_big modules."""

import importlib
import pkgutil

_PACKAGE = __name__.rpartition(".")[0] or "rolling_big"


def list_functions():
    """Return the names listed in ``__all__`` of every module in the package."""
    package = importlib.import_module(_PACKAGE)
    names = []
    for info in pkgutil.iter_modules(package.__path__):
        module = importlib.import_module(f"{_PACKAGE}.{info.name}")
        names.extend(getattr(module, "__all__", []))
    return names
