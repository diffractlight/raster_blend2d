# Repository Guidelines

## Project Structure & Module Organization

This repository packages Blend2D as a Python extension for layout rasterization.

- `src/raster_blend2d/` contains the public Python package.
- `src/raster_blend2d_native/` contains pybind11/C++ bindings.
- `extern/blend2d/` is expected to contain the Blend2D source tree.
- `extern/asmjit/` is expected to contain Blend2D's AsmJit dependency.
- `tests/` contains pytest tests.
- `docs/` contains contributor and release notes.
- `.github/workflows/` contains CI wheel-building automation.

Keep generated files in `build/`, `dist/`, `wheelhouse/`, or `_skbuild/`; these paths are ignored.

## Build, Test, and Development Commands

- `git clone --depth 1 https://github.com/blend2d/blend2d extern/blend2d` adds Blend2D.
- `git clone --depth 1 https://github.com/asmjit/asmjit extern/asmjit` adds AsmJit.
- `python -m pip install -U pip build pytest` installs local development tools.
- `python -m pip install -e .` builds and installs the extension in editable mode.
- `pytest` runs the test suite.
- `python -m build` builds source and wheel distributions.

Native builds require CMake 3.21+, a C++17 compiler, pybind11, and scikit-build-core.

## Coding Style & Naming Conventions

Use C++17 in native code and keep bindings thin. Prefer small Python-facing functions that map cleanly to layout rasterization tasks, such as `render_rects_prgb32`. Use 2-space indentation in CMake/TOML/YAML and follow the surrounding style in C++ and Python files. Name C++ files descriptively, for example `bindings.cpp` or `path_builder.cpp`.

## Testing Guidelines

Use pytest. Test files should be named `test_*.py` and should live under `tests/`. Rendering tests should prefer deterministic checks: image dimensions, pixel format, byte length, selected pixel values, hashes, or small golden fixtures. Add coverage when changing native memory ownership, pixel formats, or Blend2D context behavior.

## Commit & Pull Request Guidelines

Git history is unavailable in this checkout, so no local convention can be inferred. Use short imperative commit messages, for example `Add rectangle raster binding` or `Fix Linux wheel build`.

Pull requests should include a concise summary, build/test commands run, linked issues, and screenshots or image diffs for visual rasterization changes.

## Release Notes

See `docs/release.md` for PyPI publishing. Windows and Linux wheels are built with `cibuildwheel`; tagged releases matching `v*` publish through trusted publishing.
