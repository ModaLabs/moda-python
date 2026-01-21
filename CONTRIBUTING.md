# Contributing to Moda Python SDK

Thanks for taking the time to contribute!

## How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`uv run pytest tests/`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Development Setup

```bash
cd packages/moda
uv sync --all-extras --dev
```

## Running Tests

```bash
uv run pytest tests/ -v
```

## Code Style

We use `ruff` for linting:

```bash
uv run ruff check traceloop/
```
