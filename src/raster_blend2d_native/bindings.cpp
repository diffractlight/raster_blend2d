#include <blend2d.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
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
    double window_height_um,
    int oversampling) {
  py::sequence ring = py::reinterpret_borrow<py::sequence>(ring_obj);
  if (ring.size() < 3) {
    return;
  }

  for (py::ssize_t i = 0; i < ring.size(); ++i) {
    double x_um = point_coord(ring[i], 0);
    double y_um = point_coord(ring[i], 1);
    double x = x_um / pixel_size_x_um * static_cast<double>(oversampling);
    double y = (window_height_um - y_um) / pixel_size_y_um * static_cast<double>(oversampling);

    if (i == 0) {
      path.move_to(x, y);
    } else {
      path.line_to(x, y);
    }
  }
  path.close();
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

  int mask_width = width_px * oversampling;
  int mask_height = height_px * oversampling;

  BLImage mask(mask_width, mask_height, BL_FORMAT_A8);
  BLContext ctx(mask);
  ctx.set_comp_op(BL_COMP_OP_SRC_COPY);
  ctx.fill_all(BLRgba32(0x00000000u));
  ctx.set_comp_op(BL_COMP_OP_SRC_OVER);
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
        window_size_um[1],
        oversampling);

    if (polygon.contains("holes") && !polygon["holes"].is_none()) {
      py::sequence holes = py::reinterpret_borrow<py::sequence>(polygon["holes"]);
      for (const py::handle& hole : holes) {
        add_ring_to_path(
            path,
            hole,
            pixel_size_um[0],
            pixel_size_um[1],
            window_size_um[1],
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

  py::array_t<float> coverage({width_px, height_px});
  auto out = coverage.mutable_unchecked<2>();
  const auto* pixels = static_cast<const std::uint8_t*>(data.pixel_data);
  const double scale = 1.0 / (255.0 * static_cast<double>(oversampling) * static_cast<double>(oversampling));

  for (int x = 0; x < width_px; ++x) {
    for (int y = 0; y < height_px; ++y) {
      std::uint64_t sum = 0;
      int mask_y0 = (height_px - 1 - y) * oversampling;
      int mask_x0 = x * oversampling;

      for (int sy = 0; sy < oversampling; ++sy) {
        const auto* row = pixels + static_cast<std::size_t>(mask_y0 + sy) * static_cast<std::size_t>(data.stride);
        for (int sx = 0; sx < oversampling; ++sx) {
          sum += row[mask_x0 + sx];
        }
      }

      out(x, y) = static_cast<float>(static_cast<double>(sum) * scale);
    }
  }

  return coverage;
}

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
}
