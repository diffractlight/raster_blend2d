# Release Process

This project is intended to publish binary wheels for Windows and Linux through
GitHub Actions and trusted publishing.

## One-Time Setup

1. Add Blend2D at `extern/blend2d` as a Git submodule.
2. Add AsmJit at `extern/asmjit` as a Git submodule.
3. Create or reserve the PyPI project `raster-blend2d`.
4. Configure PyPI trusted publishing for this repository and the `wheels.yml`
   workflow.
5. Verify repository URLs in `pyproject.toml`.

Trusted publisher settings:

```text
Project name: raster-blend2d
Owner: diffractlight
Repository: raster_blend2d
Workflow: wheels.yml
Environment: pypi
```

## Local Validation

```powershell
python -m pip install -U pip build pytest
python -m pip install -e .
pytest
python -m build
```

## Publish

Create and push a version tag:

```powershell
git tag v0.1.0
git push origin v0.1.0
```

The workflow builds wheels on `ubuntu-latest` and `windows-latest`, then
publishes to PyPI only for `v*` tags.
