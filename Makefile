.PHONY: test-e2e-mocked test-e2e-db test-e2e test-e2e-up test-e2e-down test-e2e-wait-db test-e2e-regression test-predeploy

DATABASE_URL_TEST ?= postgresql+asyncpg://spx_test:spx_test_pw@localhost:5434/index_spread_lab_test
PYTHON_BIN ?= python3.11

# Resolve relative Python paths from repo root while preserving plain command names.
PYTHON_BIN_CMD := $(if $(filter /%,$(PYTHON_BIN)),$(PYTHON_BIN),$(if $(findstring /,$(PYTHON_BIN)),$(CURDIR)/$(PYTHON_BIN),$(PYTHON_BIN)))

test-e2e-up:
	docker compose -f docker-compose.test.yml up -d postgres_test

test-e2e-down:
	docker compose -f docker-compose.test.yml down

test-e2e-wait-db:
	@echo "Waiting for postgres_test to become healthy..."
	@set -e; \
	for i in $$(seq 1 30); do \
		status=$$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}starting{{end}}' index-spread-lab-pg-test 2>/dev/null || true); \
		if [ "$$status" = "healthy" ]; then \
			echo "postgres_test is healthy"; \
			exit 0; \
		fi; \
		echo "postgres_test status=$$status (attempt $$i/30)"; \
		sleep 2; \
	done; \
	echo "postgres_test did not become healthy in time"; \
	docker compose -f docker-compose.test.yml logs postgres_test; \
	exit 1

test-e2e-mocked:
	cd backend && "$(PYTHON_BIN_CMD)" -m pytest -q -m "e2e and not integration"

test-e2e-db:
	cd backend && DATABASE_URL_TEST="$(DATABASE_URL_TEST)" "$(PYTHON_BIN_CMD)" -m pytest -q -m integration

test-e2e: test-e2e-up test-e2e-wait-db test-e2e-mocked test-e2e-db

test-e2e-regression:
	@set -e; \
	trap '$(MAKE) -C "$(CURDIR)" test-e2e-down' EXIT; \
	$(MAKE) test-e2e-up; \
	$(MAKE) test-e2e-wait-db; \
	(cd backend && DATABASE_URL_TEST="$(DATABASE_URL_TEST)" "$(PYTHON_BIN_CMD)" -m pytest -q -m "integration and regression")

test-predeploy:
	@set -e; \
	trap '$(MAKE) -C "$(CURDIR)" test-e2e-down' EXIT; \
	$(MAKE) test-e2e-up; \
	$(MAKE) test-e2e-wait-db; \
	(cd backend && "$(PYTHON_BIN_CMD)" -m pytest -q -m "not integration"); \
	(cd backend && DATABASE_URL_TEST="$(DATABASE_URL_TEST)" "$(PYTHON_BIN_CMD)" -m pytest -q -m integration)
