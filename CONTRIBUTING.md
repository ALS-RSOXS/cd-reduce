# Contributing to cd-reduce

Thank you for your interest in contributing to `cd-reduce`!

## Development Setup

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/als-rsoxs/cd-reduce.git
   cd cd-reduce
   ```
2. Install dependencies using `uv` (including `dev` and `docs` dependency groups):
   ```bash
   uv sync --all-groups
   ```
3. Install `prek` hooks:
   ```bash
   prek install
   ```

## Making Changes

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes and ensure tests pass:
   ```bash
   uv run pytest
   ```
3. Ensure code quality:
   ```bash
   uv run ruff check .
   uv run ruff format --check .
   uv run ty check
   ```
4. Build the documentation strictly to verify docstrings and links:
   ```bash
   uv run mkdocs build --strict
   ```
5. Commit your changes using [Conventional Commits](https://www.conventionalcommits.org/):
   ```bash
   git commit -m "feat: add new feature"
   ```

## Commit Message Format

We use [Conventional Commits](https://www.conventionalcommits.org/). Examples:

- `feat: add new feature` — A new feature
- `fix: resolve bug in X` — A bug fix
- `docs: update guides` — Documentation changes
- `refactor: simplify Q-space binning` — Code refactoring
- `test: add tests for multislice solver` — Adding tests
- `chore: update dependencies` — Maintenance tasks

## Pull Request Process

1. Update documentation and guides if needed
2. Add unit and physics-consistency tests for new functionality
3. Ensure all tests, linting, formatting, and type checks pass
4. Submit a pull request with a clear description

## Code Style

- We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting (`line-length = 88`, double quotes)
- We use [ty](https://docs.astral.sh/ty/) for static type checking
- All public functions, methods, and dataclasses must be properly typed and documented with Google-style docstrings
- Never hard-code machine-specific or user-specific paths in library, test, or example code
