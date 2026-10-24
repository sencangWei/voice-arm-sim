"""Use Ubuntu's matching Tk extension when uv's Tk lacks Xft font rendering."""
import importlib.util
from pathlib import Path
import sys
import sysconfig


def prefer_system_tk():
    # Keep the override process-local; never replace uv or system files.
    if "_tkinter" in sys.modules or not sys.platform.startswith("linux"):
        return
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    extension = Path(f"/usr/lib/python{version}/lib-dynload/_tkinter{sysconfig.get_config_var('EXT_SUFFIX')}")
    if not extension.is_file():
        return
    spec = importlib.util.spec_from_file_location("_tkinter", extension)
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except ImportError:
        return  # Other platforms retain their own working Tk.
    sys.modules["_tkinter"] = module
