.PHONY: build babelbrain-build babelbrain-up babelbrain-check babelbrain-e2e babelbrain-transfer babelbrain-partial babelbrain-tokens babelbrain-down babelbrain-clean babelbrain-logs

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

# SF-02: a 2 GB file to the router and back, peak RSS under 500 MB on both sides
babelbrain-transfer:
	cd ../e2e && python3 -m pytest test_large_transfer.py -v -s --timeout=1800

# SF-10: site c misses round 1 and rejoins round 2, quorum 2; about 5 minutes
babelbrain-partial:
	cd ../e2e && python3 -m pytest test_partial_participation.py -v -s --timeout=1800

# SF-09: enrol every site, restart the controllers on their tokens, run 2 rounds
babelbrain-tokens:
	cd ../e2e && python3 -m pytest test_site_tokens.py -v -s --timeout=1800

babelbrain-logs:
	$(BB_COMPOSE) logs --tail=100

babelbrain-down:
	$(BB_COMPOSE) down

# Also deletes the synthetic stores and all run data
babelbrain-clean:
	$(BB_COMPOSE) down -v
