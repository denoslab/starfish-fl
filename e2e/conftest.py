"""
pytest configuration for Starfish-FL E2E tests.

Requires three Controller instances and a Router to be running:
  site-a (coordinator)   → http://localhost:8001
  site-b (participant 1) → http://localhost:8002
  site-c (participant 2) → http://localhost:8003

Start the stack with:
  cd workbench
  docker-compose -f docker-compose.e2e.yml up -d
"""
from pathlib import Path
import pytest

# Base URLs for each site's Controller web portal
BASE_A = "http://localhost:8001"
BASE_B = "http://localhost:8002"
BASE_C = "http://localhost:8003"

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def base_a():
    return BASE_A


@pytest.fixture(scope="session")
def base_b():
    return BASE_B


@pytest.fixture(scope="session")
def base_c():
    return BASE_C


@pytest.fixture(scope="session")
def fixtures_dir():
    return FIXTURES_DIR


# pytest-playwright provides the `browser` fixture (session-scoped, Chromium by default).
# The test creates its own browser contexts so each site runs in an isolated session.
