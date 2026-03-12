# Contributing

## Pull Request Workflow

Contributing to the `main` branch follows standard PR standards:

1. **Open a pull request** from your feature branch to `main`.
2. **Pass all CI checks** — linting, formatting, and tests must all pass.
3. **Request a review** from relevant moderators. If unsure, ask the repository owner.
4. **Apply requested changes** if needed.
5. **Acquire approval** from a reviewer.
6. **Merge** — this automatically triggers the workflow that syncs changes to the public repository.


## Code Style

Asparagus uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```bash
# Install ruff
pip install ruff

# Lint (check only)
ruff check asparagus/

# Lint and auto-fix
ruff check --fix asparagus/

# Format
ruff format asparagus/
```

Run tests locally with pytest:

```bash
pytest
```

## Feature Branches

Work on project-specific code on a **feature branch** branched from `main`:

```bash
git checkout main
git pull
git checkout -b my-project/feature-name
```

Keep your branch up to date by continuously merging `main` into it:

```bash
git fetch origin
git merge origin/main
```

Once your project is complete and the paper has been posted to ArXiv, open a PR to merge the code into `main` for public release.

### Sharing QOL Improvements Early

If your project includes improvements to Asparagus that don't reveal anything confidential about your project (e.g., bug fixes, performance improvements, new utility functions), we strongly encourage you to:

1. Implement the improvement on a **separate feature branch** (branched from `main`, not from your project branch).
2. Merge it into `main` separately.
3. Merge `main` back into your project branch.

This keeps all users of Asparagus up to date and makes your eventual project merge much easier.


## Documentation

Documentation lives in `docs/` and is built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/).

```bash
# Install docs dependencies
pip install -e ".[docs]"

# Preview locally
mkdocs serve

# Build static site
mkdocs build
```
