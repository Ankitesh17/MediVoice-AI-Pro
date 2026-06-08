# realtime-voice-agent-demo — realtime voice AI assistant (local demo)
# Frontend: Next.js/React/TS  |  Backend: Python/FastAPI
# Two processes; run `make dev` to start both.

.PHONY: install lint lint-fix test build dev dev-backend dev-frontend typecheck clean

install:
	cd backend && uv sync
	cd frontend && npm install

lint:
	cd backend && uv run ruff check .
	cd frontend && npm run lint

lint-fix:
	cd backend && uv run ruff check --fix .
	cd frontend && npm run lint -- --fix

typecheck:
	cd backend && uv run mypy app
	cd frontend && npm run typecheck

test:
	cd backend && uv run pytest
	cd frontend && npm test --if-present

build:
	cd frontend && npm run build

# Run both dev servers. Backend on :8000, frontend on :3000.
dev:
	@echo "Starting backend (:8000) and frontend (:3000)..."
	@$(MAKE) -j2 dev-backend dev-frontend

dev-backend:
	cd backend && uv run uvicorn app.main:app --reload --port 8000

dev-frontend:
	cd frontend && npm run dev

clean:
	find . -name 'node_modules' -type d -prune -exec rm -rf '{}' +
	find . -name '.next' -type d -prune -exec rm -rf '{}' +
	find . -name '__pycache__' -type d -prune -exec rm -rf '{}' +
	find . -name '.venv' -type d -prune -exec rm -rf '{}' +
