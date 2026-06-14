import numpy as np

from raster_blend2d import Blend2DPolygonRasterizer


polygons = [
    {
        "hull": [
            (0.0, 0.0),
            (2.0, 0.0),
            (2.0, 1.0),
            (0.0, 1.0),
        ],
        "holes": [],
    }
]

rasterizer = Blend2DPolygonRasterizer(
    pixel_size_um=(0.1, 0.1),
    oversampling=4,
)

coverage = rasterizer.rasterize(
    polygons=polygons,
    window_size_um=(3.0, 2.0),
)

print(coverage.shape)
print(coverage.dtype == np.float32)
print(coverage[0, 0])
print(coverage.max())
