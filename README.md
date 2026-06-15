# raster-blend2d

Python bindings for [Blend2D](https://blend2d.com/) focused on layout rasterization.

## Status

This project packages Blend2D as a Python extension. It expects Blend2D and
AsmJit source trees under `extern/` before native builds can succeed.

## Development

Initialize vendored dependencies:

```powershell
git submodule update --init --recursive
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
    # Optional: render in output-pixel tiles to cap peak A8 mask size.
    # tile_size_px=1024,
    # Optional: parallelize tiled flat-buffer rasterization.
    # parallel_workers=4,
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
float32 coverage. A `Blend2DPolygonRasterizer` instance caches its native A8 mask
buffer between calls; use `rasterizer.clear_cache()` to release it. Set
`tile_size_px` for very large windows to render smaller A8 mask tiles and reduce
peak memory use. `parallel_workers` parallelizes tiled `rasterize_flat()` calls;
the Python list/dict `rasterize()` path remains serial because it accesses Python
objects.

For large layouts, use flat numpy buffers to reduce Python object parsing:

```python
import numpy as np

points = np.array(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ],
    dtype=np.float64,
)
ring_offsets = np.array([0, 4], dtype=np.int64)
polygon_offsets = np.array([0, 1], dtype=np.int64)

coverage = rasterizer.rasterize_flat(
    points=points,
    ring_offsets=ring_offsets,
    polygon_offsets=polygon_offsets,
    window_size_um=(2.0, 2.0),
)
```

`ring_offsets` stores point ranges for each ring. `polygon_offsets` stores ring
ranges for each polygon; the first ring is the hull and following rings are holes.

## Benchmark

The table below compares the current implementation with the previous committed
version, `v0.1.4`, on Windows CPython 3.12. The workload is 10,000 independent
rectangles rasterized into a `700 x 700` coverage array with `pixel_size_um=0.2`.
Times are median seconds over 9 runs.

| API / configuration | Oversampling | Median time | Speedup vs v0.1.4 |
| --- | ---: | ---: | ---: |
| v0.1.4 `rasterize()` | 1 | 0.00547 s | 1.00x |
| current `rasterize()` | 1 | 0.00536 s | 1.02x |
| current `rasterize_flat()` | 1 | 0.00230 s | 2.38x |
| current `rasterize_flat(tile_size_px=256, parallel_workers=4)` | 1 | 0.00106 s | 5.17x |
| v0.1.4 `rasterize()` | 2 | 0.00745 s | 1.00x |
| current `rasterize()` | 2 | 0.00705 s | 1.06x |
| current `rasterize_flat()` | 2 | 0.00399 s | 1.87x |
| current `rasterize_flat(tile_size_px=256, parallel_workers=4)` | 2 | 0.00131 s | 5.68x |
| v0.1.4 `rasterize()` | 4 | 0.01339 s | 1.00x |
| current `rasterize()` | 4 | 0.01318 s | 1.02x |
| current `rasterize_flat()` | 4 | 0.00965 s | 1.39x |
| current `rasterize_flat(tile_size_px=256, parallel_workers=4)` | 4 | 0.00190 s | 7.05x |

The tiled parallel case benefits from flat numpy input, per-worker A8 mask reuse,
tile-level polygon bounding-box culling, and parallel tile execution. Actual
speedups depend on polygon size, spatial distribution, tile size, oversampling,
and available CPU cores.
