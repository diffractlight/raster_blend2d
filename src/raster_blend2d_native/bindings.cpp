#include <blend2d.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace py = pybind11;

struct ImageBuffer {
  int width = 0;
  int height = 0;
  int stride = 0;
  std::string format = "PRGB32";
  std::vector<std::uint8_t> pixels;
};

static void validate_size(int width, int height) {
  if (width <= 0 || height <= 0) {
    throw std::invalid_argument("width and height must be positive");
  }
}

static std::uint32_t rgba_to_argb(std::uint8_t r, std::uint8_t g, std::uint8_t b, std::uint8_t a) {
  return (static_cast<std::uint32_t>(a) << 24) |
         (static_cast<std::uint32_t>(r) << 16) |
         (static_cast<std::uint32_t>(g) << 8) |
         static_cast<std::uint32_t>(b);
}

static ImageBuffer render_rects_prgb32(
    int width,
    int height,
    const std::vector<std::array<double, 4>>& rects,
    std::array<std::uint8_t, 4> rgba,
    std::array<std::uint8_t, 4> background_rgba) {
  validate_size(width, height);

  BLImage image(width, height, BL_FORMAT_PRGB32);
  BLContext ctx(image);

  ctx.set_comp_op(BL_COMP_OP_SRC_COPY);
  ctx.set_fill_style(BLRgba32(rgba_to_argb(
      background_rgba[0], background_rgba[1], background_rgba[2], background_rgba[3])));
  ctx.fill_all();

  ctx.set_comp_op(BL_COMP_OP_SRC_OVER);
  ctx.set_fill_style(BLRgba32(rgba_to_argb(rgba[0], rgba[1], rgba[2], rgba[3])));
  for (const auto& rect : rects) {
    ctx.fill_rect(BLRect(rect[0], rect[1], rect[2], rect[3]));
  }
  ctx.end();

  BLImageData data;
  BLResult result = image.get_data(&data);
  if (result != BL_SUCCESS) {
    throw std::runtime_error("failed to read Blend2D image data");
  }

  ImageBuffer out;
  out.width = width;
  out.height = height;
  out.stride = width * 4;
  out.pixels.resize(static_cast<std::size_t>(out.stride) * static_cast<std::size_t>(height));

  const auto* src = static_cast<const std::uint8_t*>(data.pixel_data);
  for (int y = 0; y < height; ++y) {
    std::memcpy(
        out.pixels.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(out.stride),
        src + static_cast<std::size_t>(y) * static_cast<std::size_t>(data.stride),
        static_cast<std::size_t>(out.stride));
  }

  return out;
}

static int ceil_to_int(double value, const char* name) {
  if (!std::isfinite(value) || value <= 0.0) {
    throw std::invalid_argument(std::string(name) + " must be positive and finite");
  }
  double rounded = std::ceil(value);
  if (rounded > static_cast<double>(std::numeric_limits<int>::max())) {
    throw std::overflow_error(std::string(name) + " is too large");
  }
  return static_cast<int>(rounded);
}

static double point_coord(const py::handle& point, int index) {
  py::sequence seq = py::reinterpret_borrow<py::sequence>(point);
  if (seq.size() != 2) {
    throw std::invalid_argument("polygon points must be (x_um, y_um) pairs");
  }
  return py::cast<double>(seq[index]);
}

static void add_ring_to_path(
    BLPath& path,
    const py::handle& ring_obj,
    double pixel_size_x_um,
    double pixel_size_y_um,
    double window_width_um,
    int width_px,
    int tile_x_end_px,
    int tile_y_start_px,
    int oversampling) {
  py::sequence ring = py::reinterpret_borrow<py::sequence>(ring_obj);
  if (ring.size() < 3) {
    return;
  }

  for (py::ssize_t i = 0; i < ring.size(); ++i) {
    double x_um = point_coord(ring[i], 0);
    double y_um = point_coord(ring[i], 1);
    double x = (y_um / pixel_size_y_um - static_cast<double>(tile_y_start_px)) *
               static_cast<double>(oversampling);
    double y = ((window_width_um - x_um) / pixel_size_x_um -
                static_cast<double>(width_px - tile_x_end_px)) *
               static_cast<double>(oversampling);

    if (i == 0) {
      path.move_to(x, y);
    } else {
      path.line_to(x, y);
    }
  }
  path.close();
}

static void add_flat_ring_to_path(
    BLPath& path,
    const double* points,
    std::int64_t point_start,
    std::int64_t point_end,
    double pixel_size_x_um,
    double pixel_size_y_um,
    double window_width_um,
    int width_px,
    int tile_x_end_px,
    int tile_y_start_px,
    int oversampling) {
  if (point_end - point_start < 3) {
    return;
  }

  for (std::int64_t i = point_start; i < point_end; ++i) {
    double x_um = points[static_cast<std::size_t>(i) * 2u];
    double y_um = points[static_cast<std::size_t>(i) * 2u + 1u];
    double x = (y_um / pixel_size_y_um - static_cast<double>(tile_y_start_px)) *
               static_cast<double>(oversampling);
    double y = ((window_width_um - x_um) / pixel_size_x_um -
                static_cast<double>(width_px - tile_x_end_px)) *
               static_cast<double>(oversampling);

    if (i == point_start) {
      path.move_to(x, y);
    } else {
      path.line_to(x, y);
    }
  }
  path.close();
}

static unsigned sum_4_u8(const std::uint8_t* values) {
  std::uint32_t word = 0;
  std::memcpy(&word, values, sizeof(word));
  return static_cast<unsigned>(word & 0xFFu) +
         static_cast<unsigned>((word >> 8) & 0xFFu) +
         static_cast<unsigned>((word >> 16) & 0xFFu) +
         static_cast<unsigned>((word >> 24) & 0xFFu);
}

static void downsample_a8_mask_into(
    const BLImageData& data,
    float* out,
    int width_px,
    int height_px,
    int tile_x_start_px,
    int tile_y_start_px,
    int tile_width_px,
    int tile_height_px,
    int oversampling) {
  const auto* pixels = static_cast<const std::uint8_t*>(data.pixel_data);
  const double scale =
      1.0 / (255.0 * static_cast<double>(oversampling) * static_cast<double>(oversampling));

  if (oversampling == 1) {
    constexpr float kScale = 1.0f / 255.0f;
    for (int x = 0; x < tile_width_px; ++x) {
      int mask_y = tile_width_px - 1 - x;
      const auto* row =
          pixels + static_cast<std::size_t>(mask_y) * static_cast<std::size_t>(data.stride);
      auto* out_row = out +
          static_cast<std::size_t>(tile_x_start_px + x) * static_cast<std::size_t>(height_px) +
          static_cast<std::size_t>(tile_y_start_px);
      for (int y = 0; y < tile_height_px; ++y) {
        out_row[y] = static_cast<float>(row[y]) * kScale;
      }
    }
    return;
  }

  if (oversampling == 2) {
    constexpr float kScale = 1.0f / (255.0f * 4.0f);
    for (int x = 0; x < tile_width_px; ++x) {
      int mask_y0 = (tile_width_px - 1 - x) * 2;
      const auto* row0 =
          pixels + static_cast<std::size_t>(mask_y0) * static_cast<std::size_t>(data.stride);
      const auto* row1 =
          pixels + static_cast<std::size_t>(mask_y0 + 1) * static_cast<std::size_t>(data.stride);
      auto* out_row = out +
          static_cast<std::size_t>(tile_x_start_px + x) * static_cast<std::size_t>(height_px) +
          static_cast<std::size_t>(tile_y_start_px);
      for (int y = 0; y < tile_height_px; ++y) {
        int mask_x0 = y * 2;
        unsigned sum = static_cast<unsigned>(row0[mask_x0]) +
                       static_cast<unsigned>(row0[mask_x0 + 1]) +
                       static_cast<unsigned>(row1[mask_x0]) +
                       static_cast<unsigned>(row1[mask_x0 + 1]);
        out_row[y] = static_cast<float>(sum) * kScale;
      }
    }
    return;
  }

  if (oversampling == 4) {
    constexpr float kScale = 1.0f / (255.0f * 16.0f);
    for (int x = 0; x < tile_width_px; ++x) {
      int mask_y0 = (tile_width_px - 1 - x) * 4;
      auto* out_row = out +
          static_cast<std::size_t>(tile_x_start_px + x) * static_cast<std::size_t>(height_px) +
          static_cast<std::size_t>(tile_y_start_px);
      for (int y = 0; y < tile_height_px; ++y) {
        int mask_x0 = y * 4;
        unsigned sum = 0;
        for (int sy = 0; sy < 4; ++sy) {
          const auto* row =
              pixels + static_cast<std::size_t>(mask_y0 + sy) * static_cast<std::size_t>(data.stride);
          sum += sum_4_u8(row + mask_x0);
        }
        out_row[y] = static_cast<float>(sum) * kScale;
      }
    }
    return;
  }

  for (int x = 0; x < tile_width_px; ++x) {
    auto* out_row = out +
        static_cast<std::size_t>(tile_x_start_px + x) * static_cast<std::size_t>(height_px) +
        static_cast<std::size_t>(tile_y_start_px);
    for (int y = 0; y < tile_height_px; ++y) {
      std::uint64_t sum = 0;
      int mask_y0 = (tile_width_px - 1 - x) * oversampling;
      int mask_x0 = y * oversampling;

      for (int sy = 0; sy < oversampling; ++sy) {
        const auto* row =
            pixels + static_cast<std::size_t>(mask_y0 + sy) * static_cast<std::size_t>(data.stride);
        for (int sx = 0; sx < oversampling; ++sx) {
          sum += row[mask_x0 + sx];
        }
      }

      out_row[y] = static_cast<float>(static_cast<double>(sum) * scale);
    }
  }
}

static py::array_t<float> downsample_a8_mask(
    const BLImageData& data,
    int width_px,
    int height_px,
    int oversampling) {
  py::array_t<float> coverage({width_px, height_px});
  auto* out = static_cast<float*>(coverage.mutable_data());
  downsample_a8_mask_into(
      data, out, width_px, height_px, 0, 0, width_px, height_px, oversampling);
  return coverage;
}

static py::array_t<float> rasterize_polygons_a8(
    const py::sequence& polygons,
    std::array<double, 2> pixel_size_um,
    int oversampling,
    std::array<double, 2> window_size_um) {
  if (oversampling <= 0) {
    throw std::invalid_argument("oversampling must be positive");
  }
  if (!std::isfinite(pixel_size_um[0]) || !std::isfinite(pixel_size_um[1]) ||
      pixel_size_um[0] <= 0.0 || pixel_size_um[1] <= 0.0) {
    throw std::invalid_argument("pixel_size_um values must be positive and finite");
  }

  int width_px = ceil_to_int(window_size_um[0] / pixel_size_um[0], "window width in pixels");
  int height_px = ceil_to_int(window_size_um[1] / pixel_size_um[1], "window height in pixels");

  if (width_px > std::numeric_limits<int>::max() / oversampling ||
      height_px > std::numeric_limits<int>::max() / oversampling) {
    throw std::overflow_error("oversampled mask dimensions are too large");
  }

  int mask_width = height_px * oversampling;
  int mask_height = width_px * oversampling;

  BLImage mask(mask_width, mask_height, BL_FORMAT_A8);
  BLContext ctx(mask);
  ctx.set_comp_op(BL_COMP_OP_SRC_COPY);
  ctx.fill_all(BLRgba32(0x00000000u));
  ctx.set_comp_op(BL_COMP_OP_PLUS);
  ctx.set_fill_rule(BL_FILL_RULE_EVEN_ODD);

  for (const py::handle& polygon_obj : polygons) {
    py::dict polygon = py::reinterpret_borrow<py::dict>(polygon_obj);
    if (!polygon.contains("hull")) {
      throw std::invalid_argument("each polygon must contain a 'hull'");
    }

    BLPath path;
    add_ring_to_path(
        path,
        polygon["hull"],
        pixel_size_um[0],
        pixel_size_um[1],
        window_size_um[0],
        width_px,
        width_px,
        0,
        oversampling);

    if (polygon.contains("holes") && !polygon["holes"].is_none()) {
      py::sequence holes = py::reinterpret_borrow<py::sequence>(polygon["holes"]);
      for (const py::handle& hole : holes) {
        add_ring_to_path(
            path,
            hole,
            pixel_size_um[0],
            pixel_size_um[1],
            window_size_um[0],
            width_px,
            width_px,
            0,
            oversampling);
      }
    }

    ctx.fill_path(path, BLRgba32(0xFFFFFFFFu));
  }
  ctx.end();

  BLImageData data;
  BLResult result = mask.get_data(&data);
  if (result != BL_SUCCESS) {
    throw std::runtime_error("failed to read Blend2D A8 mask data");
  }

  return downsample_a8_mask(data, width_px, height_px, oversampling);
}

static py::array_t<float> rasterize_polygons_a8_flat(
    py::array_t<double, py::array::c_style | py::array::forcecast> points,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> ring_offsets,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> polygon_offsets,
    std::array<double, 2> pixel_size_um,
    int oversampling,
    std::array<double, 2> window_size_um) {
  if (oversampling <= 0) {
    throw std::invalid_argument("oversampling must be positive");
  }
  if (!std::isfinite(pixel_size_um[0]) || !std::isfinite(pixel_size_um[1]) ||
      pixel_size_um[0] <= 0.0 || pixel_size_um[1] <= 0.0) {
    throw std::invalid_argument("pixel_size_um values must be positive and finite");
  }
  if (points.ndim() != 2 || points.shape(1) != 2) {
    throw std::invalid_argument("points must be a numpy array with shape (N, 2)");
  }
  if (ring_offsets.ndim() != 1 || ring_offsets.shape(0) < 2) {
    throw std::invalid_argument("ring_offsets must be a 1D numpy array with length >= 2");
  }
  if (polygon_offsets.ndim() != 1 || polygon_offsets.shape(0) < 2) {
    throw std::invalid_argument("polygon_offsets must be a 1D numpy array with length >= 2");
  }

  int width_px = ceil_to_int(window_size_um[0] / pixel_size_um[0], "window width in pixels");
  int height_px = ceil_to_int(window_size_um[1] / pixel_size_um[1], "window height in pixels");

  if (width_px > std::numeric_limits<int>::max() / oversampling ||
      height_px > std::numeric_limits<int>::max() / oversampling) {
    throw std::overflow_error("oversampled mask dimensions are too large");
  }

  const auto point_count = static_cast<std::int64_t>(points.shape(0));
  const auto ring_count = static_cast<std::int64_t>(ring_offsets.shape(0) - 1);
  const auto polygon_count = static_cast<std::int64_t>(polygon_offsets.shape(0) - 1);
  const double* point_data = points.data();
  const std::int64_t* ring_data = ring_offsets.data();
  const std::int64_t* polygon_data = polygon_offsets.data();

  if (ring_data[0] != 0 || polygon_data[0] != 0) {
    throw std::invalid_argument("ring_offsets[0] and polygon_offsets[0] must be 0");
  }
  for (std::int64_t i = 0; i < ring_count; ++i) {
    if (ring_data[i] > ring_data[i + 1] || ring_data[i] < 0 || ring_data[i + 1] > point_count) {
      throw std::invalid_argument("ring_offsets must be monotonic and within points");
    }
  }
  for (std::int64_t i = 0; i < polygon_count; ++i) {
    if (polygon_data[i] >= polygon_data[i + 1] ||
        polygon_data[i] < 0 ||
        polygon_data[i + 1] > ring_count) {
      throw std::invalid_argument("polygon_offsets must be monotonic and each polygon needs a hull ring");
    }
  }

  int mask_width = height_px * oversampling;
  int mask_height = width_px * oversampling;

  BLImage mask(mask_width, mask_height, BL_FORMAT_A8);
  BLContext ctx(mask);
  ctx.set_comp_op(BL_COMP_OP_SRC_COPY);
  ctx.fill_all(BLRgba32(0x00000000u));
  ctx.set_comp_op(BL_COMP_OP_PLUS);
  ctx.set_fill_rule(BL_FILL_RULE_EVEN_ODD);

  for (std::int64_t polygon_index = 0; polygon_index < polygon_count; ++polygon_index) {
    std::int64_t ring_start = polygon_data[polygon_index];
    std::int64_t ring_end = polygon_data[polygon_index + 1];
    BLPath path;

    for (std::int64_t ring_index = ring_start; ring_index < ring_end; ++ring_index) {
      add_flat_ring_to_path(
          path,
          point_data,
          ring_data[ring_index],
          ring_data[ring_index + 1],
          pixel_size_um[0],
          pixel_size_um[1],
          window_size_um[0],
          width_px,
          width_px,
          0,
          oversampling);
    }

    ctx.fill_path(path, BLRgba32(0xFFFFFFFFu));
  }
  ctx.end();

  BLImageData data;
  BLResult result = mask.get_data(&data);
  if (result != BL_SUCCESS) {
    throw std::runtime_error("failed to read Blend2D A8 mask data");
  }

  return downsample_a8_mask(data, width_px, height_px, oversampling);
}

class PolygonRasterizerA8 {
 public:
  PolygonRasterizerA8(
      std::array<double, 2> pixel_size_um,
      int oversampling,
      int tile_size_px,
      int parallel_workers)
      : pixel_size_um_(pixel_size_um),
        oversampling_(oversampling),
        tile_size_px_(tile_size_px),
        parallel_workers_(parallel_workers) {
    validate_common();
  }

  py::array_t<float> rasterize(
      const py::sequence& polygons,
      std::array<double, 2> window_size_um) {
    const int width_px = ceil_to_int(window_size_um[0] / pixel_size_um_[0], "window width in pixels");
    const int height_px = ceil_to_int(window_size_um[1] / pixel_size_um_[1], "window height in pixels");
    validate_oversampled_size(width_px, height_px);

    py::array_t<float> coverage({width_px, height_px});
    auto* out = static_cast<float*>(coverage.mutable_data());

    for_each_tile(width_px, height_px, [&](int tile_x, int tile_y, int tile_width, int tile_height) {
      render_sequence_tile(polygons, window_size_um, width_px, height_px, tile_x, tile_y, tile_width, tile_height);
      BLImageData data;
      BLResult result = mask_.get_data(&data);
      if (result != BL_SUCCESS) {
        throw std::runtime_error("failed to read Blend2D A8 mask data");
      }
      downsample_a8_mask_into(
          data, out, width_px, height_px, tile_x, tile_y, tile_width, tile_height, oversampling_);
    });

    return coverage;
  }

  py::array_t<float> rasterize_flat(
      py::array_t<double, py::array::c_style | py::array::forcecast> points,
      py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> ring_offsets,
      py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> polygon_offsets,
      std::array<double, 2> window_size_um) {
    validate_flat_inputs(points, ring_offsets, polygon_offsets);

    const int width_px = ceil_to_int(window_size_um[0] / pixel_size_um_[0], "window width in pixels");
    const int height_px = ceil_to_int(window_size_um[1] / pixel_size_um_[1], "window height in pixels");
    validate_oversampled_size(width_px, height_px);

    py::array_t<float> coverage({width_px, height_px});
    auto* out = static_cast<float*>(coverage.mutable_data());
    std::vector<PolygonBounds> polygon_bounds =
        compute_flat_polygon_bounds(points, ring_offsets, polygon_offsets);

    std::vector<Tile> tiles = collect_tiles(width_px, height_px);
    if (parallel_workers_ > 1 && tiles.size() > 1) {
      rasterize_flat_tiles_parallel(
          points,
          ring_offsets,
          polygon_offsets,
          window_size_um,
          width_px,
          height_px,
          polygon_bounds,
          tiles,
          out);
    } else {
      for (const Tile& tile : tiles) {
        render_flat_tile(
            points,
            ring_offsets,
            polygon_offsets,
            window_size_um,
            width_px,
            height_px,
            polygon_bounds.data(),
            tile.x,
            tile.y,
            tile.width,
            tile.height);
        BLImageData data;
        BLResult result = mask_.get_data(&data);
        if (result != BL_SUCCESS) {
          throw std::runtime_error("failed to read Blend2D A8 mask data");
        }
        downsample_a8_mask_into(
            data, out, width_px, height_px, tile.x, tile.y, tile.width, tile.height, oversampling_);
      }
    }

    return coverage;
  }

  void clear_cache() {
    mask_.reset();
    mask_width_ = 0;
    mask_height_ = 0;
    worker_masks_.clear();
    worker_mask_widths_.clear();
    worker_mask_heights_.clear();
  }

  std::array<int, 2> cache_size() const {
    int width = mask_width_;
    int height = mask_height_;
    for (std::size_t i = 0; i < worker_mask_widths_.size(); ++i) {
      width = std::max(width, worker_mask_widths_[i]);
      height = std::max(height, worker_mask_heights_[i]);
    }
    return {width, height};
  }

 private:
  std::array<double, 2> pixel_size_um_;
  int oversampling_ = 1;
  int tile_size_px_ = 0;
  int parallel_workers_ = 1;
  BLImage mask_;
  int mask_width_ = 0;
  int mask_height_ = 0;
  std::vector<BLImage> worker_masks_;
  std::vector<int> worker_mask_widths_;
  std::vector<int> worker_mask_heights_;

  struct Tile {
    int x = 0;
    int y = 0;
    int width = 0;
    int height = 0;
  };

  struct PolygonBounds {
    double min_x = 0.0;
    double min_y = 0.0;
    double max_x = 0.0;
    double max_y = 0.0;
  };

  void validate_common() const {
    if (oversampling_ <= 0) {
      throw std::invalid_argument("oversampling must be positive");
    }
    if (!std::isfinite(pixel_size_um_[0]) || !std::isfinite(pixel_size_um_[1]) ||
        pixel_size_um_[0] <= 0.0 || pixel_size_um_[1] <= 0.0) {
      throw std::invalid_argument("pixel_size_um values must be positive and finite");
    }
    if (tile_size_px_ < 0) {
      throw std::invalid_argument("tile_size_px must be non-negative");
    }
    if (parallel_workers_ <= 0) {
      throw std::invalid_argument("parallel_workers must be positive");
    }
  }

  void validate_oversampled_size(int width_px, int height_px) const {
    if (width_px > std::numeric_limits<int>::max() / oversampling_ ||
        height_px > std::numeric_limits<int>::max() / oversampling_) {
      throw std::overflow_error("oversampled mask dimensions are too large");
    }
  }

  static void validate_flat_inputs(
      const py::array& points,
      const py::array& ring_offsets,
      const py::array& polygon_offsets) {
    if (points.ndim() != 2 || points.shape(1) != 2) {
      throw std::invalid_argument("points must be a numpy array with shape (N, 2)");
    }
    if (ring_offsets.ndim() != 1 || ring_offsets.shape(0) < 2) {
      throw std::invalid_argument("ring_offsets must be a 1D numpy array with length >= 2");
    }
    if (polygon_offsets.ndim() != 1 || polygon_offsets.shape(0) < 2) {
      throw std::invalid_argument("polygon_offsets must be a 1D numpy array with length >= 2");
    }
  }

  std::vector<PolygonBounds> compute_flat_polygon_bounds(
      py::array_t<double, py::array::c_style | py::array::forcecast> points,
      py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> ring_offsets,
      py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> polygon_offsets) const {
    const auto point_count = static_cast<std::int64_t>(points.shape(0));
    const auto ring_count = static_cast<std::int64_t>(ring_offsets.shape(0) - 1);
    const auto polygon_count = static_cast<std::int64_t>(polygon_offsets.shape(0) - 1);
    const double* point_data = points.data();
    const std::int64_t* ring_data = ring_offsets.data();
    const std::int64_t* polygon_data = polygon_offsets.data();
    validate_flat_offsets(point_count, ring_count, polygon_count, ring_data, polygon_data);

    std::vector<PolygonBounds> bounds(static_cast<std::size_t>(polygon_count));
    for (std::int64_t polygon_index = 0; polygon_index < polygon_count; ++polygon_index) {
      PolygonBounds polygon_bounds;
      polygon_bounds.min_x = std::numeric_limits<double>::infinity();
      polygon_bounds.min_y = std::numeric_limits<double>::infinity();
      polygon_bounds.max_x = -std::numeric_limits<double>::infinity();
      polygon_bounds.max_y = -std::numeric_limits<double>::infinity();

      for (std::int64_t ring_index = polygon_data[polygon_index];
           ring_index < polygon_data[polygon_index + 1];
           ++ring_index) {
        for (std::int64_t point_index = ring_data[ring_index];
             point_index < ring_data[ring_index + 1];
             ++point_index) {
          const double x_px = point_data[static_cast<std::size_t>(point_index) * 2u] / pixel_size_um_[0];
          const double y_px = point_data[static_cast<std::size_t>(point_index) * 2u + 1u] / pixel_size_um_[1];
          polygon_bounds.min_x = std::min(polygon_bounds.min_x, x_px);
          polygon_bounds.min_y = std::min(polygon_bounds.min_y, y_px);
          polygon_bounds.max_x = std::max(polygon_bounds.max_x, x_px);
          polygon_bounds.max_y = std::max(polygon_bounds.max_y, y_px);
        }
      }

      bounds[static_cast<std::size_t>(polygon_index)] = polygon_bounds;
    }

    return bounds;
  }

  static bool intersects_tile(
      const PolygonBounds& bounds,
      int tile_x,
      int tile_y,
      int tile_width,
      int tile_height) {
    return bounds.max_x > static_cast<double>(tile_x) &&
           bounds.min_x < static_cast<double>(tile_x + tile_width) &&
           bounds.max_y > static_cast<double>(tile_y) &&
           bounds.min_y < static_cast<double>(tile_y + tile_height);
  }

  template <class Callback>
  void for_each_tile(int width_px, int height_px, Callback&& callback) {
    for (const Tile& tile : collect_tiles(width_px, height_px)) {
      callback(tile.x, tile.y, tile.width, tile.height);
    }
  }

  std::vector<Tile> collect_tiles(int width_px, int height_px) const {
    std::vector<Tile> tiles;
    const int tile_size = tile_size_px_ > 0 ? tile_size_px_ : std::max(width_px, height_px);
    for (int tile_x = 0; tile_x < width_px; tile_x += tile_size) {
      const int tile_width = std::min(tile_size, width_px - tile_x);
      for (int tile_y = 0; tile_y < height_px; tile_y += tile_size) {
        const int tile_height = std::min(tile_size, height_px - tile_y);
        tiles.push_back(Tile{tile_x, tile_y, tile_width, tile_height});
      }
    }
    return tiles;
  }

  void ensure_mask(int width, int height) {
    if (mask_width_ < width || mask_height_ < height) {
      mask_ = BLImage(width, height, BL_FORMAT_A8);
      mask_width_ = width;
      mask_height_ = height;
    }
  }

  void prepare_mask(int tile_width_px, int tile_height_px) {
    ensure_mask(tile_height_px * oversampling_, tile_width_px * oversampling_);
    BLContext ctx(mask_);
    ctx.set_comp_op(BL_COMP_OP_SRC_COPY);
    ctx.fill_all(BLRgba32(0x00000000u));
    ctx.end();
  }

  BLContext begin_draw(int tile_width_px, int tile_height_px) {
    prepare_mask(tile_width_px, tile_height_px);
    BLContext ctx(mask_);
    ctx.set_comp_op(BL_COMP_OP_PLUS);
    ctx.set_fill_rule(BL_FILL_RULE_EVEN_ODD);
    ctx.clip_to_rect(BLRectI(0, 0, tile_height_px * oversampling_, tile_width_px * oversampling_));
    return ctx;
  }

  static void ensure_mask(BLImage& mask, int& mask_width, int& mask_height, int width, int height) {
    if (mask_width < width || mask_height < height) {
      mask = BLImage(width, height, BL_FORMAT_A8);
      mask_width = width;
      mask_height = height;
    }
  }

  void prepare_mask(BLImage& mask, int& mask_width, int& mask_height, int tile_width_px, int tile_height_px) {
    ensure_mask(mask, mask_width, mask_height, tile_height_px * oversampling_, tile_width_px * oversampling_);
    BLContext ctx(mask);
    ctx.set_comp_op(BL_COMP_OP_SRC_COPY);
    ctx.fill_all(BLRgba32(0x00000000u));
    ctx.end();
  }

  BLContext begin_draw(
      BLImage& mask,
      int& mask_width,
      int& mask_height,
      int tile_width_px,
      int tile_height_px) {
    prepare_mask(mask, mask_width, mask_height, tile_width_px, tile_height_px);
    BLContext ctx(mask);
    ctx.set_comp_op(BL_COMP_OP_PLUS);
    ctx.set_fill_rule(BL_FILL_RULE_EVEN_ODD);
    ctx.clip_to_rect(BLRectI(0, 0, tile_height_px * oversampling_, tile_width_px * oversampling_));
    return ctx;
  }

  void render_sequence_tile(
      const py::sequence& polygons,
      std::array<double, 2> window_size_um,
      int width_px,
      int /*height_px*/,
      int tile_x,
      int tile_y,
      int tile_width,
      int tile_height) {
    BLContext ctx = begin_draw(tile_width, tile_height);
    const int tile_x_end = tile_x + tile_width;

    for (const py::handle& polygon_obj : polygons) {
      py::dict polygon = py::reinterpret_borrow<py::dict>(polygon_obj);
      if (!polygon.contains("hull")) {
        throw std::invalid_argument("each polygon must contain a 'hull'");
      }

      BLPath path;
      add_ring_to_path(
          path,
          polygon["hull"],
          pixel_size_um_[0],
          pixel_size_um_[1],
          window_size_um[0],
          width_px,
          tile_x_end,
          tile_y,
          oversampling_);

      if (polygon.contains("holes") && !polygon["holes"].is_none()) {
        py::sequence holes = py::reinterpret_borrow<py::sequence>(polygon["holes"]);
        for (const py::handle& hole : holes) {
          add_ring_to_path(
              path,
              hole,
              pixel_size_um_[0],
              pixel_size_um_[1],
              window_size_um[0],
              width_px,
              tile_x_end,
              tile_y,
              oversampling_);
        }
      }

      ctx.fill_path(path, BLRgba32(0xFFFFFFFFu));
    }
    ctx.end();
  }

  void render_flat_tile(
      py::array_t<double, py::array::c_style | py::array::forcecast> points,
      py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> ring_offsets,
      py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> polygon_offsets,
      std::array<double, 2> window_size_um,
      int width_px,
      int /*height_px*/,
      const PolygonBounds* polygon_bounds,
      int tile_x,
      int tile_y,
      int tile_width,
      int tile_height) {
    const auto point_count = static_cast<std::int64_t>(points.shape(0));
    const auto ring_count = static_cast<std::int64_t>(ring_offsets.shape(0) - 1);
    const auto polygon_count = static_cast<std::int64_t>(polygon_offsets.shape(0) - 1);
    const double* point_data = points.data();
    const std::int64_t* ring_data = ring_offsets.data();
    const std::int64_t* polygon_data = polygon_offsets.data();

    validate_flat_offsets(point_count, ring_count, polygon_count, ring_data, polygon_data);

    BLContext ctx = begin_draw(tile_width, tile_height);
    const int tile_x_end = tile_x + tile_width;

    for (std::int64_t polygon_index = 0; polygon_index < polygon_count; ++polygon_index) {
      if (!intersects_tile(
              polygon_bounds[static_cast<std::size_t>(polygon_index)],
              tile_x,
              tile_y,
              tile_width,
              tile_height)) {
        continue;
      }
      std::int64_t ring_start = polygon_data[polygon_index];
      std::int64_t ring_end = polygon_data[polygon_index + 1];
      BLPath path;

      for (std::int64_t ring_index = ring_start; ring_index < ring_end; ++ring_index) {
        add_flat_ring_to_path(
            path,
            point_data,
            ring_data[ring_index],
            ring_data[ring_index + 1],
            pixel_size_um_[0],
            pixel_size_um_[1],
            window_size_um[0],
            width_px,
            tile_x_end,
            tile_y,
            oversampling_);
      }

      ctx.fill_path(path, BLRgba32(0xFFFFFFFFu));
    }
    ctx.end();
  }

  void render_flat_tile(
      BLImage& mask,
      int& mask_width,
      int& mask_height,
      const double* point_data,
      const std::int64_t* ring_data,
      const std::int64_t* polygon_data,
      std::int64_t polygon_count,
      std::array<double, 2> window_size_um,
      int width_px,
      int /*height_px*/,
      const PolygonBounds* polygon_bounds,
      int tile_x,
      int tile_y,
      int tile_width,
      int tile_height) {
    BLContext ctx = begin_draw(mask, mask_width, mask_height, tile_width, tile_height);
    const int tile_x_end = tile_x + tile_width;

    for (std::int64_t polygon_index = 0; polygon_index < polygon_count; ++polygon_index) {
      if (!intersects_tile(
              polygon_bounds[static_cast<std::size_t>(polygon_index)],
              tile_x,
              tile_y,
              tile_width,
              tile_height)) {
        continue;
      }
      std::int64_t ring_start = polygon_data[polygon_index];
      std::int64_t ring_end = polygon_data[polygon_index + 1];
      BLPath path;

      for (std::int64_t ring_index = ring_start; ring_index < ring_end; ++ring_index) {
        add_flat_ring_to_path(
            path,
            point_data,
            ring_data[ring_index],
            ring_data[ring_index + 1],
            pixel_size_um_[0],
            pixel_size_um_[1],
            window_size_um[0],
            width_px,
            tile_x_end,
            tile_y,
            oversampling_);
      }

      ctx.fill_path(path, BLRgba32(0xFFFFFFFFu));
    }
    ctx.end();
  }

  void rasterize_flat_tiles_parallel(
      py::array_t<double, py::array::c_style | py::array::forcecast> points,
      py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> ring_offsets,
      py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> polygon_offsets,
      std::array<double, 2> window_size_um,
      int width_px,
      int height_px,
      const std::vector<PolygonBounds>& polygon_bounds,
      const std::vector<Tile>& tiles,
      float* out) {
    const auto point_count = static_cast<std::int64_t>(points.shape(0));
    const auto ring_count = static_cast<std::int64_t>(ring_offsets.shape(0) - 1);
    const auto polygon_count = static_cast<std::int64_t>(polygon_offsets.shape(0) - 1);
    const double* point_data = points.data();
    const std::int64_t* ring_data = ring_offsets.data();
    const std::int64_t* polygon_data = polygon_offsets.data();
    validate_flat_offsets(point_count, ring_count, polygon_count, ring_data, polygon_data);

    const std::size_t worker_count = std::min<std::size_t>(
        static_cast<std::size_t>(parallel_workers_),
        tiles.size());
    worker_masks_.resize(worker_count);
    worker_mask_widths_.resize(worker_count, 0);
    worker_mask_heights_.resize(worker_count, 0);

    std::atomic<std::size_t> next_tile{0};
    std::exception_ptr first_error;
    std::mutex error_mutex;
    std::vector<std::thread> threads;
    threads.reserve(worker_count);

    {
      py::gil_scoped_release release;
      for (std::size_t worker_index = 0; worker_index < worker_count; ++worker_index) {
        threads.emplace_back([&, worker_index]() {
          try {
            while (true) {
              const std::size_t tile_index = next_tile.fetch_add(1, std::memory_order_relaxed);
              if (tile_index >= tiles.size()) {
                break;
              }

              const Tile& tile = tiles[tile_index];
              render_flat_tile(
                  worker_masks_[worker_index],
                  worker_mask_widths_[worker_index],
                  worker_mask_heights_[worker_index],
                  point_data,
                  ring_data,
                  polygon_data,
                  polygon_count,
                  window_size_um,
                  width_px,
                  height_px,
                  polygon_bounds.data(),
                  tile.x,
                  tile.y,
                  tile.width,
                  tile.height);

              BLImageData data;
              BLResult result = worker_masks_[worker_index].get_data(&data);
              if (result != BL_SUCCESS) {
                throw std::runtime_error("failed to read Blend2D A8 mask data");
              }
              downsample_a8_mask_into(
                  data,
                  out,
                  width_px,
                  height_px,
                  tile.x,
                  tile.y,
                  tile.width,
                  tile.height,
                  oversampling_);
            }
          } catch (...) {
            std::lock_guard<std::mutex> lock(error_mutex);
            if (!first_error) {
              first_error = std::current_exception();
            }
          }
        });
      }

      for (std::thread& thread : threads) {
        thread.join();
      }
    }

    if (first_error) {
      std::rethrow_exception(first_error);
    }
  }

  static void validate_flat_offsets(
      std::int64_t point_count,
      std::int64_t ring_count,
      std::int64_t polygon_count,
      const std::int64_t* ring_data,
      const std::int64_t* polygon_data) {
    if (ring_data[0] != 0 || polygon_data[0] != 0) {
      throw std::invalid_argument("ring_offsets[0] and polygon_offsets[0] must be 0");
    }
    for (std::int64_t i = 0; i < ring_count; ++i) {
      if (ring_data[i] > ring_data[i + 1] || ring_data[i] < 0 || ring_data[i + 1] > point_count) {
        throw std::invalid_argument("ring_offsets must be monotonic and within points");
      }
    }
    for (std::int64_t i = 0; i < polygon_count; ++i) {
      if (polygon_data[i] >= polygon_data[i + 1] ||
          polygon_data[i] < 0 ||
          polygon_data[i + 1] > ring_count) {
        throw std::invalid_argument("polygon_offsets must be monotonic and each polygon needs a hull ring");
      }
    }
  }
};

PYBIND11_MODULE(_core, m) {
  m.doc() = "Native Blend2D rasterization bindings";

  py::class_<ImageBuffer>(m, "ImageBuffer")
      .def_readonly("width", &ImageBuffer::width)
      .def_readonly("height", &ImageBuffer::height)
      .def_readonly("stride", &ImageBuffer::stride)
      .def_readonly("format", &ImageBuffer::format)
      .def_property_readonly("pixels", [](const ImageBuffer& image) {
        return py::bytes(reinterpret_cast<const char*>(image.pixels.data()), image.pixels.size());
      });

  py::class_<PolygonRasterizerA8>(m, "_A8Rasterizer")
      .def(
          py::init<std::array<double, 2>, int, int, int>(),
          py::arg("pixel_size_um"),
          py::arg("oversampling"),
          py::arg("tile_size_px") = 0,
          py::arg("parallel_workers") = 1)
      .def(
          "rasterize",
          &PolygonRasterizerA8::rasterize,
          py::arg("polygons"),
          py::arg("window_size_um"))
      .def(
          "rasterize_flat",
          &PolygonRasterizerA8::rasterize_flat,
          py::arg("points"),
          py::arg("ring_offsets"),
          py::arg("polygon_offsets"),
          py::arg("window_size_um"))
      .def("clear_cache", &PolygonRasterizerA8::clear_cache)
      .def_property_readonly("cache_size", &PolygonRasterizerA8::cache_size);

  m.def(
      "render_rects_prgb32",
      &render_rects_prgb32,
      py::arg("width"),
      py::arg("height"),
      py::arg("rects"),
      py::arg("rgba") = std::array<std::uint8_t, 4>{255, 255, 255, 255},
      py::arg("background_rgba") = std::array<std::uint8_t, 4>{0, 0, 0, 0},
      "Rasterize rectangles and return raw Blend2D PRGB32 pixels.");

  m.def(
      "_rasterize_polygons_a8",
      &rasterize_polygons_a8,
      py::arg("polygons"),
      py::arg("pixel_size_um"),
      py::arg("oversampling"),
      py::arg("window_size_um"),
      "Rasterize polygons through a Blend2D A8 mask and return float32 coverage[x, y].");

  m.def(
      "_rasterize_polygons_a8_flat",
      &rasterize_polygons_a8_flat,
      py::arg("points"),
      py::arg("ring_offsets"),
      py::arg("polygon_offsets"),
      py::arg("pixel_size_um"),
      py::arg("oversampling"),
      py::arg("window_size_um"),
      "Rasterize flat numpy polygon buffers through a Blend2D A8 mask.");
}
