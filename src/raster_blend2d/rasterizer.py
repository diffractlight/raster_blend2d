"""Polygon rasterization API compatible with the Skia rasterizer."""

from __future__ import annotations

from ._core import _rasterize_polygons_a8


class Blend2DPolygonRasterizer:
    """Rasterize layout polygons to float32 coverage using a Blend2D A8 mask."""

    def __init__(self, pixel_size_um: tuple[float, float], oversampling: int = 1) -> None:
        if len(pixel_size_um) != 2:
            raise ValueError("pixel_size_um must be a 2-tuple")
        if pixel_size_um[0] <= 0 or pixel_size_um[1] <= 0:
            raise ValueError("pixel_size_um values must be positive")
        if oversampling <= 0:
            raise ValueError("oversampling must be positive")

        self.pixel_size_um = (float(pixel_size_um[0]), float(pixel_size_um[1]))
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
