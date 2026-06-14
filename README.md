# raster-blend2d

Python bindings for [Blend2D](https://blend2d.com/) focused on layout rasterization.

## Status

This project is an initial packaging and binding skeleton. It expects Blend2D source at
`extern/blend2d` before native builds can succeed.

## Development

Add Blend2D and AsmJit as vendored dependencies:

```powershell
git clone --depth 1 https://github.com/blend2d/blend2d extern/blend2d
git clone --depth 1 https://github.com/asmjit/asmjit extern/asmjit
```

Build and test locally:

```powershell
python -m pip install -U pip build pytest
python -m pip install -e .
pytest
```

Build a wheel:

```powershell
python -m build
```

## Polygon Rasterization API

```python
from raster_blend2d import Blend2DPolygonRasterizer

rasterizer = Blend2DPolygonRasterizer(
    pixel_size_um=(0.01, 0.02),
    oversampling=4,
)

polygons = [
    {
        "hull": [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        "holes": [],
    }
]

coverage = rasterizer.rasterize(
    polygons=polygons,
    window_size_um=(2.0, 2.0),
)

assert coverage.dtype == "float32"
assert coverage.shape == (200, 100)
assert coverage[0, 0] >= 0.0
```

`coverage[x, y]` uses a lower-left window origin. Internally, polygons are rendered
to a Blend2D `A8` mask at the requested oversampling rate and downsampled to
float32 coverage.
