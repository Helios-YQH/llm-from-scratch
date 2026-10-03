import importlib.metadata

try:
    __version__ = importlib.metadata.version("lm-basics")
except importlib.metadata.PackageNotFoundError:
    pass
