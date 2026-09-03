"""SimCortex package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("simcortex")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
