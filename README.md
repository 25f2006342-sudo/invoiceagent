# Invoice Action Agent (A2A 1.0)

FastAPI starter implementation for the supplied invoice-agent contract. It proposes actions, stores task state, binds accepted receipts to stored proposals, scopes tasks to bearer-token principals, and persists idempotency records.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env  # Windows; configure values and export them in your shell
uvicorn app:app --host 0.0.0.0 --port 8000
```

The agent card is at `/.well-known/agent-card.json`; API routes are under `/a2a/`.

## Deploy

1. Push this directory to a private or public Git repository.
2. Create a Render web service from the repository and use the included `render.yaml`.
3. Set `A2A_BASE_URL` to the exact final public HTTPS URL ending in `/a2a/`.
4. Set a long random `A2A_BEARER_TOKENS` secret and an AI provider key. Never commit secrets.
5. Confirm the service has a persistent disk mounted at `/var/data`; SQLite on an ephemeral filesystem loses state on restart.
6. Submit the exact base URL ending in `/a2a/` to the grader.

## Important hardening before relying on it

This is a starter, not a payment system. No external payment is initiated by this code; the only execution record is a receipt for an accepted grader result. Before production, implement the grader's precise concurrent cancel/result atomicity with a single database transaction and conditional state update, add strict package/result schemas, enforce the exact `application/a2a+json` response contract for all errors, and test with the real challenge harness. SQLite is suitable for a single instance; use PostgreSQL for multi-instance deployment. Configure a provider/model that can process the full 12-package batch within the 45-second request deadline.
