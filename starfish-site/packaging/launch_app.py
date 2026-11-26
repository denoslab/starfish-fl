"""Starts Starfish Site, the windowed program that PyInstaller builds from this file."""
import sys

from starfish_site.app.main import main

sys.exit(main())
