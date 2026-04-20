import importlib
import sys
from pathlib import Path

# The groove-signal src/ package needs src.node and src.relay from the
# parent monorepo. Splice them into the groove-signal src namespace so
# both are importable under "src.*".
_parent_src = Path(__file__).resolve().parent.parent.parent / "src"

for sub in ("node", "relay"):
    subdir = _parent_src / sub
    if subdir.is_dir():
        pkg_name = f"src.{sub}"
        if pkg_name not in sys.modules:
            spec = importlib.util.spec_from_file_location(
                pkg_name, subdir / "__init__.py",
                submodule_search_locations=[str(subdir)],
            )
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                sys.modules[pkg_name] = mod
                spec.loader.exec_module(mod)
