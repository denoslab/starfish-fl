# Starfish Controller

> **Note**: This is a component of the Starfish federated learning platform. For the complete system overview and setup instructions, see the [main README](../README.md).

A federated learning (FL) system that is friendly to users with diverse backgrounds,
for instance, in healthcare. This is the Controller component.

## Overview

A **Controller** is installed on every site participating in federated learning.
With the Controller running, a Site can act as either a **Coordinator** or a **Participant**.

For information about the overall Starfish architecture, see the [main documentation](../README.md).

## Installation Options

### Option 1: Mono Repo Setup (Recommended)

If you're working with the complete Starfish mono repo, use the workbench for a unified setup:

```shell
cd ../workbench
make build
make up
```

See the [main README](../README.md) and [workbench documentation](../workbench/README.md) for details.

### Option 2: Standalone Docker Compose

For standalone controller deployment or development:

This is the easiest way to get started. Docker Compose will set up both the application, Redis cache, and Celery workers.

#### Prerequisites

- [Docker](https://docs.docker.com/engine/install/)
- [Docker Compose](https://docs.docker.com/compose/install/)

#### Setup Instructions

1. **Configure environment variables**
   
   Copy `.env.example` to `.env` and update the values:
   ```shell
   cp .env.example .env
   ```
   
   Update the following in `.env`:
   ```properties
   SITE_UID=<generate-unique-uuid>
   ROUTER_URL=http://your-router-url:8000/starfish/api/v1
   ROUTER_USERNAME=your_username
   ROUTER_PASSWORD=your_password
   ```

2. **Build the images**
   ```shell
   docker-compose build
   ```

3. **Start the services**
   ```shell
   docker-compose up -d
   ```

4. **Run database migrations**
   ```shell
   docker exec -it starfish-controller poetry run python3 manage.py migrate
   ```

5. **Access the application**
   
   Open your browser and navigate to: http://localhost:8001/

6. **Stop the services**
   ```shell
   docker-compose stop
   ```

   To stop and remove containers:
   ```shell
   docker-compose down
   ```

### Option 3: Local Development (Without Docker)

This method gives you more control but requires manual setup of Redis and Python environment.

#### Prerequisites

- Python 3.10.10
- Redis server
- [Poetry](https://python-poetry.org/) for dependency management
- [pyenv](https://github.com/pyenv/pyenv) (recommended for Python version management on macOS/Linux)
- [pyenv-win](https://github.com/pyenv-win/pyenv-win) (recommended for Python version management on Windows)

#### Setup Instructions

1. **Install pyenv** (if not already installed)

   **macOS:**
   ```shell
   brew update
   brew install pyenv
   ```

   **Linux:**

   Follow the instructions here:
   ```shell
   https://github.com/pyenv/pyenv?tab=readme-ov-file#installation
   ```

   **Windows:**
   
   Follow the instructions here:
   ```powershell
   https://github.com/pyenv/pyenv?tab=readme-ov-file#installation
   ```
   
   After installation, restart your PowerShell/Command Prompt.

2. **Install and configure Python 3.10.10**
   
   **macOS/Linux:**
   ```shell
   pyenv install 3.10.10
   pyenv local 3.10.10
   ```
   
   **Windows:**
   ```powershell
   pyenv install 3.10.10
   pyenv local 3.10.10
   ```

3. **Create and activate virtual environment**
   
   **macOS/Linux:**
   ```shell
   python -m venv .venv
   source .venv/bin/activate
   ```
   
   **Windows (PowerShell):**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

4. **Install Poetry** (if not already installed)
   
   **macOS/Linux:**
   ```shell
   curl -sSL https://install.python-poetry.org | python3 -
   ```
   
   **Windows (PowerShell):**
   ```powershell
   (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
   ```
   
   After installation, add Poetry to your PATH if it's not already added.

5. **Install project dependencies**
   ```shell
   poetry install
   ```

6. **Install and start Redis**
   
   **macOS:**
   ```shell
   brew install redis
   brew services start redis
   ```
   
   **Linux (Ubuntu/Debian):**
   ```shell
   sudo apt-get install redis-server
   sudo systemctl start redis
   ```
   
   **Windows:**
   
   Download Redis from [https://redis.io/download](https://redis.io/download) or use WSL.

7. **Configure environment variables**
   
   Copy `.env.example` to `.env` and update the values:
   ```shell
   cp .env.example .env
   ```
   
   Update Redis connection settings for local development:
   ```properties
   SITE_UID=<generate-unique-uuid>
   ROUTER_URL=http://your-router-url:8000/starfish/api/v1
   ROUTER_USERNAME=your_username
   ROUTER_PASSWORD=your_password
   CELERY_BROKER_URL=redis://localhost:6379
   CELERY_RESULT_BACKEND=redis://localhost:6379
   REDIS_HOST=localhost
   REDIS_PORT=6379
   REDIS_DB=0
   ```

8. **Run database migrations**
   ```shell
   python3 manage.py migrate
   ```
   
   **Note:** On Windows, you might need to use `python` instead of `python3`:
   ```powershell
   python manage.py migrate
   ```

9. **Start Celery workers** (in separate terminals)
   
   **Terminal 1 - Celery Beat (Scheduler):**
   ```shell
   celery -A starfish beat -l DEBUG
   ```
   
   **Terminal 2 - Run Worker:**
   ```shell
   celery -A starfish worker -l DEBUG -Q starfish.run
   ```
   
   **Terminal 3 - Processor Worker:**
   ```shell
   celery -A starfish worker -l DEBUG --concurrency=1 -Q starfish.processor
   ```

10. **Start the development server** (in a new terminal)
    ```shell
    python3 manage.py runserver
    ```
    
    **Windows:**
    ```powershell
    python manage.py runserver
    ```

11. **Access the application**
    
    Open your browser and navigate to: http://localhost:8001/

#### Deactivate virtual environment

**macOS/Linux:**
```shell
deactivate
```

**Windows:**
```powershell
deactivate
```

## Development Tasks

### Running Tests

Run the test suite:
```shell
python3 manage.py test
```

**Windows:**
```powershell
python manage.py test
```

### Code Formatting

Format Python code using autopep8:
```shell
autopep8 --exclude='*/migrations/*' --in-place --recursive ./starfish/
```

### Managing Dependencies

**Add a new dependency:**
```shell
poetry add <package_name>
```

**Add a development dependency:**
```shell
poetry add --group=dev <package_name>
```

**Remove a dependency:**
```shell
poetry remove <package_name>
```

## Production Deployment

### Prerequisites

- Access to the git repository
- Docker and Docker Compose installed
- Access to a running Starfish Router instance
- Redis server (managed by Docker Compose)
- Internet access
- Properly configured firewall and network settings

### Configuration

Before deployment, configure the following files:

#### 1. `.env` file
Configure application settings by copying `.env.example` to `.env`:
```properties
SITE_UID=<generate-unique-uuid>              # Unique identifier for this site
ROUTER_URL=http://your-router:8000/starfish/api/v1  # URL to Starfish Router
ROUTER_USERNAME=your_username                         # Router username
ROUTER_PASSWORD=your_password             # Router password
CELERY_BROKER_URL=redis://redis:6379         # Redis connection for Celery
CELERY_RESULT_BACKEND=redis://redis:6379     # Redis backend for results
REDIS_HOST=redis                              # Redis host (service name)
REDIS_PORT=6379                               # Redis port
REDIS_DB=0                                    # Redis database number
```

**Important:** 
- Generate a unique `SITE_UID` for each deployment (use `uuidgen` or similar tool)
- Update `ROUTER_URL` to point to your Starfish Router instance
- Use secure credentials for `ROUTER_USERNAME` and `ROUTER_PASSWORD`
- Set `REDIS_HOST=redis` (the service name in Docker Compose)

**Also Note:**
- Service Port: 8001 is by default and it is forwarded from the docker container. Please update if it has conflict with your existing service
- Volumes: The service will be running inside the docker container, but the mounted volumes will keep the intermedia files(logs and models). /starfish-controller/local by default, please update it if needed.
- Database: The redis is used as a cache storage and pub-sub service. By default, /opt/redis/data will store the cache database data as the mount volume.

### Network Security

Ensure proper network configuration:

- **Firewall:** Configure to allow access from trusted sources (other sites and users)
- **IP Whitelist:** Restrict access to specific IP addresses if needed
- **Port Access:** Ensure the service port (default 8001) is accessible from authorized networks
- **Router Connectivity:** Ensure the controller can reach the Routing Server
- **Additional Security:** Consider using:
  - Reverse proxy (e.g., Nginx)
  - SSL/TLS certificates
  - VPN for site-to-site communication

### Deployment Steps

1. **Configure environment**
   
   Copy and update the `.env` file:
   ```shell
   cp .env.example .env
   # Edit .env with your settings
   ```

2. **Build the images**
   ```shell
   docker-compose build
   ```

3. **Start the services**
   ```shell
   docker-compose up -d
   ```

4. **Run database migrations**
   ```shell
   docker exec -it starfish-controller poetry run python3 manage.py migrate
   ```

5. **Verify the deployment**
   
   Visit http://your-server-ip:8001/ (replace with your actual server address and port)

6. **Register the site with the Router**
   
   The controller will automatically attempt to register with the Router using the credentials in `.env`.
   Check the logs to verify successful registration:
   ```shell
   docker-compose logs -f starfish-controller
   ```

### Maintenance

#### View logs
```shell
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f starfish-controller
docker-compose logs -f controller-run-worker
```

#### Restart services
```shell
docker-compose restart
```

#### Stop services
```shell
docker-compose stop
```

#### Stop and remove containers
```shell
docker-compose down
```

#### Update the application
```shell
git pull
docker-compose build
docker-compose up -d
```

### Backup and Recovery

#### Backup database and artifacts
```shell
# Backup SQLite database
docker cp starfish-controller:/app/db.sqlite3 ./backup_db_$(date +%Y%m%d).sqlite3
```

## Common Issues:
A common issue when using docker compose is:
```md
ERROR: for redis  Cannot start service redis: Ports are not available: exposing port TCP 0.0.0.0:6379 -> 0.0.0.0:0: listen tcp 0.0.0.0:6379: bind: address already in use
```
It happens because redis running in docker container conflicts with the local redis already running and occupying port 6379.
Solution: Simply stop the local redis service.
```shell
sudo systemctl stop redis
```