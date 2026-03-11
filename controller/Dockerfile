FROM python:3.10

WORKDIR /app

# Prevent Python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Disable Poetry virtualenv creation to install directly in system python
ENV POETRY_VIRTUALENVS_CREATE=false

# Install runtime dependencies (including R for R-based FL tasks)
# python:3.10 includes gcc, python headers, etc.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    libmagic1 \
    netcat-traditional \
    libpq-dev \
    dos2unix \
    r-base-core \
    r-recommended \
    gfortran \
    libnlopt-dev \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Install R packages required by R-based tasks
RUN Rscript -e "install.packages(c('jsonlite', 'survival', 'mice'), repos='https://cloud.r-project.org')"

# Install Poetry first to leverage cache
RUN pip3 install --no-cache-dir poetry

COPY poetry.lock pyproject.toml /app/

# Install project dependencies
RUN poetry install --no-root --no-cache

# Copy the rest of the project files
COPY . /app/

RUN chmod +x ./entrypoint.sh
RUN dos2unix ./entrypoint.sh

ENTRYPOINT [ "./entrypoint.sh" ]
