import os
import sys


def get_project_root() -> str:
    """Return the project root directory (directory containing ``src``)."""
    cwd = os.getcwd()
    while not os.path.isdir(os.path.join(cwd, "src")):
        parent = os.path.dirname(cwd)
        if parent == cwd:
            break
        cwd = parent
    return cwd


def add_project_root_to_sys_path() -> str:
    """Add project root to ``sys.path`` and return it."""
    root = get_project_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    return root
