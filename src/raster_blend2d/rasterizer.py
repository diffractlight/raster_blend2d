"""Polygon rasterization API compatible with the Skia rasterizer."""

from __future__ import annotations

from numbers import Real

from ._core import _A8Rasterizer


def _normalize_mask_format(mask_format: str) -> str:
    value = str(mask_format).lower()
    if value not in {"a8", "prgb32"}:
        raise ValueError("mask_format must be 'a8' or 'prgb32'")
    return value


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

    def __init__(
        self,
        pixel_size_um: float | tuple[float, float],
        oversampling: int = 1,
        tile_size_px: int | None = None,
        parallel_workers: int = 1,
        snap_to_pixel_grid: bool = True,
        mask_format: str = "a8",
    ) -> None:
        if oversampling <= 0:
            raise ValueError("oversampling must be positive")
        if tile_size_px is not None and tile_size_px <= 0:
            raise ValueError("tile_size_px must be positive when provided")
        if parallel_workers <= 0:
            raise ValueError("parallel_workers must be positive")

        self.pixel_size_um = _normalize_pixel_size_um(pixel_size_um)
        self.oversampling = int(oversampling)
        self.tile_size_px = int(tile_size_px or 0)
        self.parallel_workers = int(parallel_workers)
        self.snap_to_pixel_grid = bool(snap_to_pixel_grid)
        self.mask_format = _normalize_mask_format(mask_format)
        self._rasterizer = _A8Rasterizer(
            self.pixel_size_um,
            self.oversampling,
            self.tile_size_px,
            self.parallel_workers,
            self.snap_to_pixel_grid,
            self.mask_format == "prgb32",
        )

    def rasterize(self, polygons, window_size_um: tuple[float, float]):
        if len(window_size_um) != 2:
            raise ValueError("window_size_um must be a 2-tuple")
        if window_size_um[0] <= 0 or window_size_um[1] <= 0:
            raise ValueError("window_size_um values must be positive")

        return self._rasterizer.rasterize(
            polygons,
            (float(window_size_um[0]), float(window_size_um[1])),
        )

    def rasterize_flat(
        self,
        points,
        ring_offsets,
        polygon_offsets,
        window_size_um: tuple[float, float],
    ):
        """Rasterize flat numpy polygon buffers.

        points is shaped (N, 2). ring_offsets has length R + 1 and stores point
        ranges for each ring. polygon_offsets has length P + 1 and stores ring
        ranges for each polygon; the first ring of each polygon is the hull and
        following rings are holes.
        """
        if len(window_size_um) != 2:
            raise ValueError("window_size_um must be a 2-tuple")
        if window_size_um[0] <= 0 or window_size_um[1] <= 0:
            raise ValueError("window_size_um values must be positive")

        return self._rasterizer.rasterize_flat(
            points,
            ring_offsets,
            polygon_offsets,
            (float(window_size_um[0]), float(window_size_um[1])),
        )

    def clear_cache(self) -> None:
        """Release the cached native A8 mask buffer."""
        self._rasterizer.clear_cache()

    @property
    def cache_size(self) -> tuple[int, int]:
        """Return cached native A8 mask size as (width, height)."""
        return tuple(self._rasterizer.cache_size)
