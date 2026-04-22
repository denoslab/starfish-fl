# ============================================================
# Stage 1: Build R packages (compile-heavy, discarded later)
# ============================================================
FROM python:3.10-slim AS r-builder

# Temporarily disabled: R build dependencies
# RUN apt-get update \
#     && apt-get install -y --no-install-recommends \
#     r-base-core \
#     gfortran \
#     libnlopt-dev \
#     cmake \
#     liblapack-dev \
#     libblas-dev \
#     make \
#     g++ \
#     zlib1g-dev \
#     pkg-config \
#     && rm -rf /var/lib/apt/lists/*

# Install R packages into a dedicated library path so we can copy just these
# Fail the build if any package fails to install
# ENV R_LIBS_USER=/opt/R/libs
# RUN mkdir -p $R_LIBS_USER \
#     && Rscript -e "\
#     pkgs <- c('jsonlite', 'survival', 'mice'); \
#     install.packages(pkgs, lib='$R_LIBS_USER', repos='https://cloud.r-project.org', Ncpus=parallel::detectCores()); \
#     for (p in pkgs) { if (!requireNamespace(p, lib.loc='$R_LIBS_USER', quietly=TRUE)) stop(paste('Package', p, 'failed to install')) }"

# Strip R packages: remove static libs, docs, help, and demo files
# RUN find $R_LIBS_USER -name '*.a' -delete \
#     && find $R_LIBS_USER -name '*.o' -delete \
#     && find $R_LIBS_USER -type d -name doc -exec rm -rf {} + 2>/dev/null || true \
#     && find $R_LIBS_USER -type d -name html -exec rm -rf {} + 2>/dev/null || true \
#     && find $R_LIBS_USER -type d -name help -exec rm -rf {} + 2>/dev/null || true \
#     && find $R_LIBS_USER -type d -name demo -exec rm -rf {} + 2>/dev/null || true \
#     && find $R_LIBS_USER -type d -name examples -exec rm -rf {} + 2>/dev/null || true

# ============================================================
# Stage 2: Runtime image
# ============================================================
FROM python:3.10-slim

WORKDIR /app

# Prevent Python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Disable Poetry virtualenv creation to install directly in system python
ENV POETRY_VIRTUALENVS_CREATE=false

# Temporarily disabled R runtime libraries:
# r-base-core, liblapack3, libblas3, libnlopt0, libgfortran5
# Install only runtime dependencies (no compilers)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    libmagic1 \
    netcat-traditional \
    libpq-dev \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-compiled R packages from builder stage
# ENV R_LIBS_USER=/opt/R/libs
# COPY --from=r-builder /opt/R/libs /opt/R/libs

# Enable UNet/image segmentation support (adds TensorFlow + CUDA libs, ~3 GB)
ARG INSTALL_UNET=false

# Install Poetry and Python dependencies (cached unless lock file changes)
RUN pip3 install --no-cache-dir poetry

COPY poetry.lock pyproject.toml /app/

# Regenerate lock if pyproject.toml changed, then install
RUN poetry lock \
    && if [ "$INSTALL_UNET" = "true" ]; then \
      poetry install --no-root --no-cache --without dev; \
    else \
      poetry install --no-root --no-cache --without dev,unet; \
    fi \
    && pip3 uninstall -y poetry virtualenv pip \
    && rm -rf /usr/local/lib/python3.10/site-packages/poetry \
              /usr/local/lib/python3.10/site-packages/virtualenv \
    && find /usr/local/lib/python3.10/site-packages -type d -name tests -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local/lib/python3.10/site-packages -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Copy the rest of the project files
COPY . /app/

RUN chmod +x ./entrypoint.sh \
    && dos2unix ./entrypoint.sh

ENTRYPOINT [ "./entrypoint.sh" ]
