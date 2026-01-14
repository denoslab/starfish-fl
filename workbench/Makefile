.PHONY: build 

COMPOSE = docker-compose
# Paths updated for mono repo structure
ROUTER_PATH = ../router
CONTROLLER_PATH = ../controller

build: router controller

router:
	$(COMPOSE) build router

controller:
	$(COMPOSE) build controller

up:
	${COMPOSE} up -d

down:
	${COMPOSE} down

stop:
	${COMPOSE} stop