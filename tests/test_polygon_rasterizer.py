import numpy as np
import pytest


def test_pixel_size_accepts_scalar_for_square_pixels():
    try:
        from raster_blend2d import Blend2DPolygonRasterizer
    except ImportError as exc:
        pytest.skip(f"native extension is not built in this checkout: {exc}")

    rasterizer = Blend2DPolygonRasterizer(pixel_size_um=0.5, oversampling=2)
    assert rasterizer.pixel_size_um == (0.5, 0.5)

    coverage = rasterizer.rasterize(
        polygons=[
            {
                "hull": [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
                "holes": [],
            }
        ],
        window_size_um=(1.0, 1.0),
    )

    assert coverage.shape == (2, 2)
    assert coverage.dtype == np.float32


def test_pixel_size_accepts_tuple_for_rectangular_pixels():
    try:
        from raster_blend2d import Blend2DPolygonRasterizer
    except ImportError as exc:
        pytest.skip(f"native extension is not built in this checkout: {exc}")

    rasterizer = Blend2DPolygonRasterizer(pixel_size_um=(0.5, 1.0), oversampling=2)
    assert rasterizer.pixel_size_um == (0.5, 1.0)

    coverage = rasterizer.rasterize(
        polygons=[
            {
                "hull": [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
                "holes": [],
            }
        ],
        window_size_um=(1.0, 1.0),
    )

    assert coverage.shape == (2, 1)
    assert coverage.dtype == np.float32


@pytest.mark.parametrize("pixel_size_um", [0.0, -1.0, (1.0,), (1.0, 0.0), object()])
def test_pixel_size_rejects_invalid_values(pixel_size_um):
    try:
        from raster_blend2d import Blend2DPolygonRasterizer
    except ImportError as exc:
        pytest.skip(f"native extension is not built in this checkout: {exc}")

    with pytest.raises(ValueError):
        Blend2DPolygonRasterizer(pixel_size_um=pixel_size_um)


def test_parallel_workers_rejects_invalid_values():
    try:
        from raster_blend2d import Blend2DPolygonRasterizer
    except ImportError as exc:
        pytest.skip(f"native extension is not built in this checkout: {exc}")

    with pytest.raises(ValueError):
        Blend2DPolygonRasterizer(pixel_size_um=1.0, parallel_workers=0)


def test_snap_to_pixel_grid_defaults_to_enabled():
    try:
        from raster_blend2d import Blend2DPolygonRasterizer
    except ImportError as exc:
        pytest.skip(f"native extension is not built in this checkout: {exc}")

    rasterizer = Blend2DPolygonRasterizer(pixel_size_um=1.0)
    assert rasterizer.snap_to_pixel_grid is True
    assert rasterizer.mask_format == "a8"


def test_snap_to_pixel_grid_can_be_disabled():
    try:
        from raster_blend2d import Blend2DPolygonRasterizer
    except ImportError as exc:
        pytest.skip(f"native extension is not built in this checkout: {exc}")

    y1 = 0.18 + 0.02
    polygons = [{"hull": [(0.01, 0.02), (0.04, 0.02), (0.04, y1), (0.01, y1)], "holes": []}]

    enabled = Blend2DPolygonRasterizer(
        pixel_size_um=0.02,
        oversampling=1,
        snap_to_pixel_grid=True,
    ).rasterize(polygons=polygons, window_size_um=(0.5, 0.5))
    disabled = Blend2DPolygonRasterizer(
        pixel_size_um=0.02,
        oversampling=1,
        snap_to_pixel_grid=False,
    ).rasterize(polygons=polygons, window_size_um=(0.5, 0.5))

    assert enabled[0, 9] > disabled[0, 9]
    assert enabled[0, 9] - disabled[0, 9] == pytest.approx(1.0 / 255.0)


def test_mask_format_accepts_prgb32():
    try:
        from raster_blend2d import Blend2DPolygonRasterizer
    except ImportError as exc:
        pytest.skip(f"native extension is not built in this checkout: {exc}")

    rasterizer = Blend2DPolygonRasterizer(pixel_size_um=1.0, mask_format="prgb32")
    coverage = rasterizer.rasterize(
        polygons=[
            {
                "hull": [(1.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0)],
                "holes": [],
            }
        ],
        window_size_um=(3.0, 2.0),
    )

    assert rasterizer.mask_format == "prgb32"
    assert coverage.shape == (3, 2)
    assert coverage[1, 0] == pytest.approx(1.0)
    assert coverage[0, 0] == pytest.approx(0.0)


def test_mask_format_rejects_invalid_values():
    try:
        from raster_blend2d import Blend2DPolygonRasterizer
    except ImportError as exc:
        pytest.skip(f"native extension is not built in this checkout: {exc}")

    with pytest.raises(ValueError):
        Blend2DPolygonRasterizer(pixel_size_um=1.0, mask_format="rgba")


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


def test_polygon_rasterizer_non_square_window_keeps_xy_orientation():
    try:
        from raster_blend2d import Blend2DPolygonRasterizer
    except ImportError as exc:
        pytest.skip(f"native extension is not built in this checkout: {exc}")

    rasterizer = Blend2DPolygonRasterizer(pixel_size_um=(1.0, 1.0), oversampling=2)
    coverage = rasterizer.rasterize(
        polygons=[
            {
                "hull": [(2.0, 1.0), (3.0, 1.0), (3.0, 2.0), (2.0, 2.0)],
                "holes": [],
            }
        ],
        window_size_um=(3.0, 2.0),
    )

    assert coverage.shape == (3, 2)
    assert coverage[2, 1] == pytest.approx(1.0)
    assert coverage[0, 0] == pytest.approx(0.0)
    assert coverage[2, 0] == pytest.approx(0.0)


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


def test_polygon_rasterizer_reuses_a8_mask_cache():
    try:
        from raster_blend2d import Blend2DPolygonRasterizer
    except ImportError as exc:
        pytest.skip(f"native extension is not built in this checkout: {exc}")

    rasterizer = Blend2DPolygonRasterizer(pixel_size_um=1.0, oversampling=2)
    polygons = [
        {
            "hull": [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)],
            "holes": [],
        }
    ]

    rasterizer.rasterize(polygons=polygons, window_size_um=(4.0, 3.0))
    first_cache_size = rasterizer.cache_size
    rasterizer.rasterize(polygons=polygons, window_size_um=(4.0, 3.0))

    assert rasterizer.cache_size == first_cache_size
    rasterizer.clear_cache()
    assert rasterizer.cache_size == (0, 0)


def test_tiled_rasterizer_matches_full_mask_sequence_api():
    try:
        from raster_blend2d import Blend2DPolygonRasterizer
    except ImportError as exc:
        pytest.skip(f"native extension is not built in this checkout: {exc}")

    polygons = [
        {
            "hull": [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)],
            "holes": [
                [(1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (1.0, 2.0)],
            ],
        },
        {
            "hull": [(3.0, 2.0), (5.0, 2.0), (5.0, 5.0), (3.0, 5.0)],
            "holes": [],
        },
    ]

    full = Blend2DPolygonRasterizer(pixel_size_um=1.0, oversampling=2).rasterize(
        polygons=polygons,
        window_size_um=(5.0, 5.0),
    )
    tiled = Blend2DPolygonRasterizer(pixel_size_um=1.0, oversampling=2, tile_size_px=2).rasterize(
        polygons=polygons,
        window_size_um=(5.0, 5.0),
    )

    np.testing.assert_allclose(tiled, full, atol=1.0 / 255.0)


def test_flat_polygon_input_matches_sequence_api():
    try:
        from raster_blend2d import Blend2DPolygonRasterizer
    except ImportError as exc:
        pytest.skip(f"native extension is not built in this checkout: {exc}")

    rasterizer = Blend2DPolygonRasterizer(pixel_size_um=(1.0, 1.0), oversampling=4)
    polygons = [
        {
            "hull": [(0.0, 0.0), (3.0, 0.0), (3.0, 3.0), (0.0, 3.0)],
            "holes": [
                [(1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (1.0, 2.0)],
            ],
        }
    ]

    sequence_coverage = rasterizer.rasterize(polygons=polygons, window_size_um=(3.0, 3.0))
    flat_coverage = rasterizer.rasterize_flat(
        points=np.array(
            [
                [0.0, 0.0],
                [3.0, 0.0],
                [3.0, 3.0],
                [0.0, 3.0],
                [1.0, 1.0],
                [2.0, 1.0],
                [2.0, 2.0],
                [1.0, 2.0],
            ],
            dtype=np.float64,
        ),
        ring_offsets=np.array([0, 4, 8], dtype=np.int64),
        polygon_offsets=np.array([0, 2], dtype=np.int64),
        window_size_um=(3.0, 3.0),
    )

    assert flat_coverage.shape == (3, 3)
    assert flat_coverage.dtype == np.float32
    np.testing.assert_allclose(flat_coverage, sequence_coverage)


def test_tiled_rasterizer_matches_full_mask_flat_api():
    try:
        from raster_blend2d import Blend2DPolygonRasterizer
    except ImportError as exc:
        pytest.skip(f"native extension is not built in this checkout: {exc}")

    points = np.array(
        [
            [0.0, 0.0],
            [4.0, 0.0],
            [4.0, 4.0],
            [0.0, 4.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [2.0, 2.0],
            [1.0, 2.0],
            [3.0, 2.0],
            [5.0, 2.0],
            [5.0, 5.0],
            [3.0, 5.0],
        ],
        dtype=np.float64,
    )
    ring_offsets = np.array([0, 4, 8, 12], dtype=np.int64)
    polygon_offsets = np.array([0, 2, 3], dtype=np.int64)

    full = Blend2DPolygonRasterizer(pixel_size_um=1.0, oversampling=2).rasterize_flat(
        points=points,
        ring_offsets=ring_offsets,
        polygon_offsets=polygon_offsets,
        window_size_um=(5.0, 5.0),
    )
    tiled = Blend2DPolygonRasterizer(pixel_size_um=1.0, oversampling=2, tile_size_px=2).rasterize_flat(
        points=points,
        ring_offsets=ring_offsets,
        polygon_offsets=polygon_offsets,
        window_size_um=(5.0, 5.0),
    )

    np.testing.assert_allclose(tiled, full, atol=1.0 / 255.0)


def test_parallel_tiled_rasterizer_matches_single_thread_flat_api():
    try:
        from raster_blend2d import Blend2DPolygonRasterizer
    except ImportError as exc:
        pytest.skip(f"native extension is not built in this checkout: {exc}")

    points = np.array(
        [
            [0.0, 0.0],
            [6.0, 0.0],
            [6.0, 3.0],
            [0.0, 3.0],
            [2.0, 1.0],
            [4.0, 1.0],
            [4.0, 2.0],
            [2.0, 2.0],
            [1.0, 3.0],
            [5.0, 3.0],
            [5.0, 6.0],
            [1.0, 6.0],
        ],
        dtype=np.float64,
    )
    ring_offsets = np.array([0, 4, 8, 12], dtype=np.int64)
    polygon_offsets = np.array([0, 2, 3], dtype=np.int64)

    single_thread = Blend2DPolygonRasterizer(
        pixel_size_um=0.5,
        oversampling=2,
        tile_size_px=3,
        parallel_workers=1,
    ).rasterize_flat(
        points=points,
        ring_offsets=ring_offsets,
        polygon_offsets=polygon_offsets,
        window_size_um=(6.0, 6.0),
    )
    parallel = Blend2DPolygonRasterizer(
        pixel_size_um=0.5,
        oversampling=2,
        tile_size_px=3,
        parallel_workers=4,
    ).rasterize_flat(
        points=points,
        ring_offsets=ring_offsets,
        polygon_offsets=polygon_offsets,
        window_size_um=(6.0, 6.0),
    )

    np.testing.assert_allclose(parallel, single_thread, atol=1.0 / 255.0)


def test_flat_polygon_input_accepts_float32_and_int32_arrays():
    try:
        from raster_blend2d import Blend2DPolygonRasterizer
    except ImportError as exc:
        pytest.skip(f"native extension is not built in this checkout: {exc}")

    rasterizer = Blend2DPolygonRasterizer(pixel_size_um=0.5, oversampling=1)
    coverage = rasterizer.rasterize_flat(
        points=np.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            dtype=np.float32,
        ),
        ring_offsets=np.array([0, 4], dtype=np.int32),
        polygon_offsets=np.array([0, 1], dtype=np.int32),
        window_size_um=(1.0, 1.0),
    )

    assert coverage.shape == (2, 2)
    assert coverage.dtype == np.float32
    assert coverage[0, 0] == pytest.approx(1.0)
    assert coverage[1, 1] == pytest.approx(1.0)


def test_flat_polygon_input_rejects_invalid_offsets():
    try:
        from raster_blend2d import Blend2DPolygonRasterizer
    except ImportError as exc:
        pytest.skip(f"native extension is not built in this checkout: {exc}")

    rasterizer = Blend2DPolygonRasterizer(pixel_size_um=1.0)
    with pytest.raises(ValueError):
        rasterizer.rasterize_flat(
            points=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
            ring_offsets=np.array([0, 4], dtype=np.int64),
            polygon_offsets=np.array([0, 1], dtype=np.int64),
            window_size_um=(1.0, 1.0),
        )
