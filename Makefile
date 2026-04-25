SEED       ?= 31312
EPISODES   ?= 30000
DATA_DIR   ?= ./data

N_REPLICAS  = 2

VENV   = .venv
PYTHON = $(VENV)/bin/python
PIP    = $(VENV)/bin/pip

.PHONY: setup run clean

setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip --timeout 120 -q
	$(PIP) install -r sim/requirements.txt --timeout 120 -q
	$(PIP) install -r botify/requirements.txt --timeout 120 -q
	cd botify && docker compose down -v --remove-orphans 2>/dev/null || true
	cd botify && docker compose up -d --build --force-recreate \
	    --scale recommender=$(N_REPLICAS)
	@echo "Waiting for service to become ready..."
	@for i in $$(seq 1 18); do \
	    STATUS=$$(curl -s -o /dev/null -w "%{http_code}" \
	        http://localhost:5001/ 2>/dev/null || echo 000); \
	    if [ "$$STATUS" = "200" ]; then \
	        echo "Service is ready (attempt $$i)"; break; \
	    fi; \
	    echo "Attempt $$i: HTTP $$STATUS, retrying in 5s..."; \
	    sleep 5; \
	done
	@curl -sf http://localhost:5001/ \
	    || (cd botify && docker compose logs && exit 1)

run:
	cd sim && echo "n" | ../$(PYTHON) -m sim.run \
	    --episodes $(EPISODES) \
	    --config   config/env.yml \
	    single --recommender remote --seed $(SEED)
	mkdir -p $(DATA_DIR)
	$(PYTHON) script/dataclient.py --recommender $(N_REPLICAS) \
	    log2local $(DATA_DIR)
	$(PYTHON) analyze_ab.py \
	    --data   $(DATA_DIR) \
	    --output $(DATA_DIR)/ab_result.json

clean:
	cd botify && docker compose down -v --remove-orphans 2>/dev/null || true
	rm -rf $(VENV)
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
