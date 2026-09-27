.PHONY: build babelbrain-build babelbrain-up babelbrain-check babelbrain-e2e babelbrain-down babelbrain-clean babelbrain-logs

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

# ── BabelBrain FL profile, see compose/babelbrain.yaml ───────────────────────
BB_COMPOSE = docker compose -p starfish-babelbrain -f compose/babelbrain.yaml
BB_SITES = controller-a-processor-worker controller-b-processor-worker controller-c-processor-worker

babelbrain-build:
	$(BB_COMPOSE) build

babelbrain-up: babelbrain-build
	$(BB_COMPOSE) up -d --wait

# REDIS_DB=15 because importing starfish clears the run cache in its Redis DB.
# Fails if a site image has anthropic installed, loads agent code, or cannot
# read its own store, or if that store is writable
babelbrain-check:
	@for s in $(BB_SITES); do \
		echo "== $$s"; \
		out=$$($(BB_COMPOSE) exec -T -e REDIS_DB=15 $$s python3 -m starfish.controller.tasks.babel_brain_fno.probe --require-no-anthropic --site-store 2>/dev/null); rc=$$?; \
		echo "$$out" | tail -1; \
		[ $$rc -eq 0 ] || exit 1; \
	done

# Acceptance run through the router API; needs babelbrain-up first
babelbrain-e2e:
	cd ../e2e && python3 -m pytest test_babelbrain_workflow.py -v --timeout=300

babelbrain-logs:
	$(BB_COMPOSE) logs --tail=100

babelbrain-down:
	$(BB_COMPOSE) down

# Also deletes the synthetic stores and all run data
babelbrain-clean:
	$(BB_COMPOSE) down -v
