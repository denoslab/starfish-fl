"""Starts the starfish-site command, the console program that PyInstaller builds from this file."""
import sys

from starfish_site.cli import main

sys.exit(main())
