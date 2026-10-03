import importlib.metadata

try:
    __version__ = importlib.metadata.version("lm-systems")
except importlib.metadata.PackageNotFoundError:
    pass
