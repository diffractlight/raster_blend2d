import numpy as np
import pytest


def test_polygon_rasterizer_shape_dtype_and_lower_left_origin():
    try:
        from raster_blend2d import Blend2DPolygonRasterizer
    except ImportError as exc:
        pytest.skip(f"native extension is not built in this checkout: {exc}")

    rasterizer = Blend2DPolygonRasterizer(pixel_size_um=(1.0, 1.0), oversampling=4)
    coverage = rasterizer.rasterize(
        polygons=[
            {
                "hull": [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
                "holes": [],
            }
        ],
        window_size_um=(2.0, 2.0),
    )

    assert coverage.shape == (2, 2)
    assert coverage.dtype == np.float32
    assert coverage[0, 0] == pytest.approx(1.0)
    assert coverage[0, 1] == pytest.approx(0.0)
    assert coverage[1, 0] == pytest.approx(0.0)


def test_polygon_rasterizer_holes_use_a8_mask_coverage():
    try:
        from raster_blend2d import Blend2DPolygonRasterizer
    except ImportError as exc:
        pytest.skip(f"native extension is not built in this checkout: {exc}")

    rasterizer = Blend2DPolygonRasterizer(pixel_size_um=(1.0, 1.0), oversampling=4)
    coverage = rasterizer.rasterize(
        polygons=[
            {
                "hull": [(0.0, 0.0), (3.0, 0.0), (3.0, 3.0), (0.0, 3.0)],
                "holes": [
                    [(1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (1.0, 2.0)],
                ],
            }
        ],
        window_size_um=(3.0, 3.0),
    )

    assert coverage[1, 1] == pytest.approx(0.0)
    assert coverage[0, 0] == pytest.approx(1.0)
    assert coverage[2, 2] == pytest.approx(1.0)
