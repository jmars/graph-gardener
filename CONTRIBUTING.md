# Contributing

Contributions are welcome — bug fixes, improved prompts, or documentation.

## Development Setup

```bash
git clone https://github.com/jmars/graph-gardener.git
cd graph-gardener

python -m venv venv
source venv/bin/activate    # or venv\Scripts\activate on Windows

pip install -e ".[dev]"
```

## Running Tests

```bash
pytest
```

With coverage:

```bash
pytest --cov=graph_gardener
```

## Linting

```bash
pip install ruff
ruff check src/ tests/
```

## Project Structure

```
src/graph_gardener/
├── __init__.py      # Package metadata
├── __main__.py      # CLI entry point
├── llm.py           # Provider-agnostic LLM client
└── gardener.py      # Core graph maintenance logic
```

## License

This project is licensed under the [MIT License](LICENSE). By contributing, you agree that your contributions will be licensed under the same license.
