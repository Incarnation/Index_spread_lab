.PHONY: test-e2e-mocked test-e2e-db test-e2e test-e2e-up test-e2e-down test-e2e-regression test-predeploy

DATABASE_URL_TEST ?= postgresql+asyncpg://spx_test:spx_test_pw@localhost:5434/index_spread_lab_test
PYTHON_BIN ?= python3.11

# Resolve relative Python paths from repo root while preserving plain command names.
PYTHON_BIN_CMD := $(if $(filter /%,$(PYTHON_BIN)),$(PYTHON_BIN),$(if $(findstring /,$(PYTHON_BIN)),$(CURDIR)/$(PYTHON_BIN),$(PYTHON_BIN)))

test-e2e-up:
	docker compose -f docker-compose.test.yml up -d postgres_test

test-e2e-down:
	docker compose -f docker-compose.test.yml down

test-e2e-mocked:
	cd backend && "$(PYTHON_BIN_CMD)" -m pytest -q -m "e2e and not integration"

test-e2e-db:
	cd backend && DATABASE_URL_TEST="$(DATABASE_URL_TEST)" "$(PYTHON_BIN_CMD)" -m pytest -q -m integration

test-e2e: test-e2e-up test-e2e-mocked test-e2e-db

test-e2e-regression:
	@set -e; \
	trap '$(MAKE) -C "$(CURDIR)" test-e2e-down' EXIT; \
	$(MAKE) test-e2e-up; \
	(cd backend && DATABASE_URL_TEST="$(DATABASE_URL_TEST)" "$(PYTHON_BIN_CMD)" -m pytest -q -m "integration and regression")

test-predeploy:
	@set -e; \
	trap '$(MAKE) -C "$(CURDIR)" test-e2e-down' EXIT; \
	$(MAKE) test-e2e-up; \
	(cd backend && "$(PYTHON_BIN_CMD)" -m pytest -q -m "not integration"); \
	(cd backend && DATABASE_URL_TEST="$(DATABASE_URL_TEST)" "$(PYTHON_BIN_CMD)" -m pytest -q -m integration)
