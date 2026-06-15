"""Polygon rasterization API compatible with the Skia rasterizer."""

from __future__ import annotations

from numbers import Real

from ._core import _rasterize_polygons_a8


def _normalize_pixel_size_um(pixel_size_um) -> tuple[float, float]:
    if isinstance(pixel_size_um, Real):
        value = float(pixel_size_um)
        if value <= 0:
            raise ValueError("pixel_size_um must be positive")
        return (value, value)

    try:
        if len(pixel_size_um) != 2:
            raise ValueError("pixel_size_um must be a positive number or a 2-tuple")
        pixel_x = float(pixel_size_um[0])
        pixel_y = float(pixel_size_um[1])
    except TypeError as exc:
        raise ValueError("pixel_size_um must be a positive number or a 2-tuple") from exc

    if pixel_x <= 0 or pixel_y <= 0:
        raise ValueError("pixel_size_um values must be positive")
    return (pixel_x, pixel_y)


class Blend2DPolygonRasterizer:
    """Rasterize layout polygons to float32 coverage using a Blend2D A8 mask."""

    def __init__(self, pixel_size_um: float | tuple[float, float], oversampling: int = 1) -> None:
        if oversampling <= 0:
            raise ValueError("oversampling must be positive")

        self.pixel_size_um = _normalize_pixel_size_um(pixel_size_um)
        self.oversampling = int(oversampling)

    def rasterize(self, polygons, window_size_um: tuple[float, float]):
        if len(window_size_um) != 2:
            raise ValueError("window_size_um must be a 2-tuple")
        if window_size_um[0] <= 0 or window_size_um[1] <= 0:
            raise ValueError("window_size_um values must be positive")

        return _rasterize_polygons_a8(
            polygons,
            self.pixel_size_um,
            self.oversampling,
            (float(window_size_um[0]), float(window_size_um[1])),
        )
