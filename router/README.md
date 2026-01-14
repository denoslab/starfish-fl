# Starfish Routing Server

> **Note**: This is a component of the Starfish federated learning platform. For the complete system overview and setup instructions, see the [main README](../README.md).

A federated learning (FL) system that is friendly to users with diverse backgrounds,
for instance, in healthcare. This is the Routing Server (RS) component.

## Overview

The project architecture diagram is shown in the [main README](../README.md).

### Routing Server (RS) Responsibilities

A **Routing Server (RS)** has two main responsibilities:

1. **Persistent Layer**: Providing a persistent layer for administrative data to facilitate smooth FL processes. RS maintains global records of users, sites, projects, and runs.

2. **Message Forwarding**: Forwarding messages between participants and coordinators in a project.

Note that RS does not do model aggregation - the Coordinator does this.
Currently, Sites exchange information with RS through polling.
Message payloads can have end-to-end encryption, and RS will not be able to read the message payloads.
Private key exchanges between sites will be done securely.

For more details about Sites, Controllers, Projects, Tasks, and Runs, see the [main documentation](../README.md).

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

For standalone router deployment or development:

This is the easiest way to get started. Docker Compose will set up both the application and PostgreSQL database.

### Prerequisites

- [Docker](https://docs.docker.com/engine/install/)
- [Docker Compose](https://docs.docker.com/compose/install/)

#### Setup Instructions

1. **Build the images**
   ```shell
   docker-compose build
   ```

2. **Start the services**
   ```shell
   docker-compose up -d
   ```

3. **Make database migrations**
   ```shell
   docker exec -it starfish-router poetry run python3 manage.py makemigrations
   ```

4. **Run database migrations**
   ```shell
   docker exec -it starfish-router poetry run python3 manage.py migrate
   ```

5. **Create a superuser** (first time only)
   ```shell
   docker exec -it starfish-router poetry run python3 manage.py createsuperuser
   ```

6. **Access the application**
   
   Open your browser and navigate to: http://localhost:8000/starfish/api/v1/

7. **Stop the services**
   ```shell
   docker-compose stop
   ```

   To stop and remove containers:
   ```shell
   docker-compose down
   ```

### Option 3: Local Development (Without Docker)

This method gives you more control but requires manual setup of PostgreSQL and Python environment.

#### Prerequisites

- Python 3.10.10
- PostgreSQL database
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

6. **Configure database connection**
   
   Update the `.env` file with your local PostgreSQL credentials:
   ```properties
   DATABASE_NAME=starfish-router
   DATABASE_USER=postgres
   DATABASE_PASSWORD=your_password
   DATABASE_HOST=localhost
   DATABASE_PORT=5432
   ```

7. **Create the database**
   
   **macOS/Linux:**
   
   If PostgreSQL is configured with peer authentication, you may need to specify the host:
   ```shell
   psql -h localhost -U postgres -c "CREATE DATABASE \"starfish-router\";"
   ```
   
   **Windows:**
   
   Using PowerShell (if PostgreSQL bin is in PATH):
   ```powershell
   & "C:\Program Files\PostgreSQL\13\bin\psql.exe" -U postgres -c "CREATE DATABASE \"starfish-router\";"
   ```
   
   Or use pgAdmin (GUI tool) to create a database named `starfish-router`.

8. **Run database migrations**
   ```shell
   python3 manage.py migrate
   ```
   
   **Note:** On Windows, you might need to use `python` instead of `python3`:
   ```powershell
   python manage.py migrate
   ```

9. **Create a superuser**
   ```shell
   python3 manage.py createsuperuser
   ```
   
   **Windows:**
   ```powershell
   python manage.py createsuperuser
   ```

10. **Start the development server**
    ```shell
    python3 manage.py runserver
    ```
    
    **Windows:**
    ```powershell
    python manage.py runserver
    ```

11. **Access the application**
    
    Open your browser and navigate to: http://localhost:8000/starfish/api/v1/

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

### Running Background Jobs

Execute a scheduled job manually:
```shell
python3 manage.py runjob <job_name>
```

Example:
```shell
python3 manage.py runjob check_site_status
```

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
- PostgreSQL database (managed by Docker Compose)
- Internet access
- Properly configured firewall and network settings

### Configuration

Before deployment, configure the following files:

#### 1. `docker-compose.yml`
- **Service Port:** Default is `8000`. Update if there's a conflict with existing services.
- **Volumes:** Mounted volumes preserve logs and model files:
  - Application artifacts: `/starfish/artifacts` (configurable)
  - Database data: `/var/lib/postgresql/data` (configurable)
- **Database credentials:** Must match those in `.env` file

#### 2. `.env` file
Configure application settings:
```properties
DEBUG=False                          # Set to False for production
SECRET_KEY=<generate-strong-key>     # Generate a secure secret key
DATABASE_NAME=starfish-router
DATABASE_USER=postgres
DATABASE_PASSWORD=<secure-password>  # Use a strong password
DATABASE_HOST=postgres               # Service name from docker-compose.yml
DATABASE_PORT=5432
```

**Important:** 
- Set `DEBUG=False` in production
- Use a strong, unique `SECRET_KEY`
- Use a secure database password
- Set `DATABASE_HOST=postgres` (the service name in Docker Compose)

### Network Security

Ensure proper network configuration:

- **Firewall:** Configure to allow access only from trusted sources
- **IP Whitelist:** Restrict access to specific IP addresses (e.g., starfish-controller instances)
- **Port Access:** Ensure the service port (default 8000) is accessible from authorized networks only
- **Additional Security:** Consider using:
  - Reverse proxy (e.g., Nginx)
  - SSL/TLS certificates
  - Third-party security services (e.g., Cloudflare)

### Deployment Steps

1. **Build the images**
   ```shell
   docker-compose build
   ```

2. **Start the services**
   ```shell
   docker-compose up -d
   ```

3. **Make database migrations**
   ```shell
   docker exec -it starfish-router poetry run python3 manage.py makemigrations
   ```

4. **Run database migrations**
   ```shell
   docker exec -it starfish-router poetry run python3 manage.py migrate
   ```

5. **Create a superuser** (first time only)
   ```shell
   docker exec -it starfish-router poetry run python3 manage.py createsuperuser
   ```
   
   **Important:** Save the username and password securely for configuring starfish-controller.

7. **Verify the deployment**
   
   Visit http://your-server-ip:8000/starfish/api/v1/ (replace with your actual server address and port)

### Maintenance

#### View logs
```shell
docker-compose logs -f starfish-router
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

#### Backup database
```shell
docker exec -it starfish-router pg_dump -U postgres starfish-router > backup_$(date +%Y%m%d).sql
```

#### Restore database
```shell
cat backup_file.sql | docker exec -i starfish-router psql -U postgres starfish-router
```

## CI/CD Pipeline Documentation
- **[CI/CD Pipeline Documentation](CI_CD_PIPELINE.md)** - Learn about automated testing and deployment