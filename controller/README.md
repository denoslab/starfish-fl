# Starfish Controller

![Project Logo](docs/images/starfish.png)

A federated learning (FL) system that is friendly to users with diverse backgrounds,
for instance, in healthcare. This repo is the Controller component.

## Overview

A **Controller** will be installed on every site. With the Controller running,
a Site can act as either a **Coordinator** or a **Participant**.

## Developers

### Environment

#### Mac

##### Prerequisites

###### brew

We are going to use [brew](https://brew.sh/) to install some tools. If you don't have brew installed on your Mac, run
the following command:

```shell
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

###### Pyenv

We would suggest to use [pyenv](https://github.com/pyenv/pyenv) to manage and switch multiple versions of Python.
To install follow steps below.

```shell
brew update
brew install pyenv
```

If this didn't work for you try these [fixes](https://github.com/pyenv/pyenv#homebrew-in-macos).

To test if this was installed correctly try running the following in your terminal:

```shell
pyenv versions
```

We are going to be using Python version 3.10.10, so let's get that installed.

```shell
pyenv install 3.10.10
```

###### Poetry

We are going to use [poetry](https://python-poetry.org/) to manage dependencies.

Run the following command in your terminal to install poetry:

```shell
curl -sSL https://install.python-poetry.org | python3 -
```

#### Windows

Todo

#### Linux

Todo

### Development

#### With Docker

##### Build docker image

```shell
docker build -t starfish-controller:latest .
```

##### Start

```shell
docker run -it -p 8000:8000 docker.io/library/starfish-controller:latest
```

#### Without Docker

##### Create a virtual environment

```shell
python3 -m venv venv
source venv/bin/activate
```

##### Install dependencies

```shell
poetry install
```

##### Database Migration

```shell
python3 manage.py migrate
```

##### Start Development Server

```shell
python3 manage.py runserver
```

##### De-active virtual environment

Type `deactivate` in your terminal

#### Code Format

Run the following command to format python code in starfish-router directory

```shell
autopep8 --exclude='*/migrations/*' --in-place --recursive ./starfish/
```

### Production Deployment

#### Prerequisites

1. Access to the git repository
2. Docker and Docker Compose is installed
3. Access to Internet

#### Configuration

Please refer to the `docker-compose.yml` and `.env.example` file. The `docker-compose.yml` defines dependencies and
middleware configs.
Change `.env.example` to `.env` and it defines configs of the starfish-controller application.

* Service Port: `8001` is by default and it is forwarded from the docker container. Please update if it has conflict
  with your existing service
* Volumes: The service will be running inside the docker container, but the mounted volumes will keep the intermedia
  files(logs and models). `/starfish-controller/local` by default, please update it if needed.
* Database: The redis is used as a cache storage and pub-sub service. By default, `/opt/redis/data` will store the cache
  database data as the mount
  volume.
* SITE_UID: Please update this environment if the starfish-controller runs in every new environment.
* ROUTER_URL: Please update this environment to the url of the starfish-router. 
* ROUTER_USERNAME and ROUTER_PASSWORD: The user is created when initializing the starfish-router.

#### To Start

1. Build images

```shell
docker-compose build
```

2. Run services under this repo folder

```shell
docker-compose up -d
```

3. Testing the connection. Visit http://localhost:8001/starfish/api/v1/ if the service port is configured as `8001`.
