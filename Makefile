.PHONY: dev test test-api test-frontend build

dev:
	./scripts/dev.sh

test: test-api test-frontend

test-api:
	pytest -q

test-frontend:
	npm --prefix frontend test -- --run

build:
	npm --prefix frontend run build
