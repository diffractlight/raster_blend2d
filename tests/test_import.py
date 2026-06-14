def test_render_rects_prgb32_smoke():
    try:
        import raster_blend2d
    except ImportError as exc:
        import pytest

        pytest.skip(f"native extension is not built in this checkout: {exc}")
        return

    image = raster_blend2d.render_rects_prgb32(
        16,
        16,
        [(1.0, 2.0, 4.0, 5.0)],
        (255, 0, 0, 255),
        (0, 0, 0, 0),
    )

    assert image.width == 16
    assert image.height == 16
    assert image.stride == 64
    assert image.format == "PRGB32"
    assert len(image.pixels) == 16 * 16 * 4
