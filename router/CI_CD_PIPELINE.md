# CI/CD Pipeline

## Overview

GitHub Actions automatically tests and deploys code on every push/pull request.

## Jobs

### Job 1: Setup and Tests
- Builds Docker images
- Starts services with docker-compose
- Tests API endpoint responds
- Runs 4 automated tests
- Uses PostgreSQL container

### Job 2: Publish (main branch only)
- Publishes Docker image to GitHub Container Registry
- Only runs if Jobs 1 & 2 pass

## Tests

**Total 4 tests** in `starfish/router/tests.py`:

1. **Database Connection** - Verifies PostgreSQL connectivity
2. **Migrations Applied** - Ensures schema is up-to-date
3. **Site Model** - Tests site creation
4. **Project Model** - Tests project creation

## Controller CI

The controller CI pipeline (`.github/workflows/controller-tests.yml`) additionally:
- Installs R (`r-base-core`, `r-recommended`) and system libraries (`gfortran`, `libnlopt-dev`, `cmake`, `liblapack-dev`, `libblas-dev`)
- Installs R packages: `jsonlite`, `survival`, `mice`
- Runs all controller tests including R-based task tests

## Running Tests Locally

```bash
# Run all tests
python3 manage.py test

# Run specific test
python3 manage.py test starfish.router.tests.DatabaseConnectionTest
```