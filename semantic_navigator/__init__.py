"""Top-level package for semantic_navigator."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("semantic-navigator")
except PackageNotFoundError:
    __version__ = "uninstalled"

__author__ = "Eva Maxfield Brown"
__email__ = "evamxb@uw.edu"
