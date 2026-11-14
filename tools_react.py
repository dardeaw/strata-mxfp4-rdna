#!/usr/bin/env python3
"""Root forwarder for tools/tools_react.py."""
import runpy
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent / "tools"
sys.path.insert(0, str(TOOLS_DIR))
runpy.run_path(str(TOOLS_DIR / "tools_react.py"), run_name="__main__")
