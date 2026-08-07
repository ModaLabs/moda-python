# AGENTS.md

Guidance for AI coding agents working in this repository.

## What this repo is

Official Python SDKs for [Moda](https://moda.dev) — the continual learning layer for AI agents. Moda turns production agent traces into validated improvements for your agent harness.

- Docs: https://docs.moda.dev
- SDK setup: https://docs.moda.dev/ingestion/moda-sdk
- Auth: https://moda.dev/auth.md

## Layout

- `packages/moda` — core SDK (`pip install moda-ai`): OpenTelemetry-native ingestion with automatic conversation threading
- `packages/moda-openai` / `packages/moda-anthropic` — provider auto-instrumentation
- `packages/moda-claude-agent-sdk` — Claude Agent SDK instrumentation

## Working in this repo

- Toolchain: `uv` per package (`uv sync --all-groups` inside a package dir), orchestrated by nx (`npm ci`, then `npx nx run <package>:install|lint|test`)
- Lint: `uv run ruff check .` per package
- Tests: `npx nx run <package>:test` (pytest under the hood); provider tests use recorded cassettes — do not hit live APIs
- Conventional-commit PR titles are enforced by CI
- Never commit API keys; the SDK reads `MODA_API_KEY` from the environment
