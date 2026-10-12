// abfFlattenInPlace stores the UVs as surface channels; saving the surface
// (what vc_flatten --uv-only does) must write them instead of throwing on a
// two-channel imwrite.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/QuadSurface.hpp"
#include "vc/flattening/ABFFlattening.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <cmath>
#include <filesystem>
#include <random>

namespace fs = std::filesystem;

namespace {

fs::path tmpDir(const std::string& tag)
{
    std::mt19937_64 rng(std::random_device{}());
    auto p = fs::temp_directory_path() / ("vc_abf_uv_" + tag + "_" + std::to_string(rng()));
    fs::create_directories(p);
    return p;
}

cv::Mat_<cv::Vec3f> curvedStrip(int rows, int cols)
{
    cv::Mat_<cv::Vec3f> p(rows, cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const float t = c * 20.f / 1200.f;
            p(r, c) = cv::Vec3f(3000.f + 1200.f * std::cos(t), 3000.f + 1200.f * std::sin(t), 5000.f + r * 20.f);
        }
    }
    return p;
}

} // namespace

TEST_CASE("abfFlattenInPlace: the UV channels of a flattened surface can be saved and read back")
{
    QuadSurface surface(curvedStrip(24, 60), cv::Vec2f(0.05f, 0.05f));
    REQUIRE(vc::abfFlattenInPlace(surface));

    const fs::path dir = tmpDir("save") / "seg";
    // vc_flatten --uv-only writes to a new directory with saveEdited(), i.e. saveAsNew().
    CHECK_NOTHROW(surface.saveAsNew(dir, true));

    REQUIRE(fs::exists(dir / "u.tif"));
    REQUIRE(fs::exists(dir / "v.tif"));
    const cv::Mat u = cv::imread((dir / "u.tif").string(), cv::IMREAD_UNCHANGED);
    const cv::Mat v = cv::imread((dir / "v.tif").string(), cv::IMREAD_UNCHANGED);
    REQUIRE(u.type() == CV_32FC1);
    REQUIRE(v.type() == CV_32FC1);
    CHECK(u.rows == 24);
    CHECK(u.cols == 60);

    // The written values are the stored channels.
    const cv::Mat cu = surface.channel("u");
    REQUIRE(cu.type() == CV_32FC1);
    CHECK(cv::norm(u, cu, cv::NORM_INF) == doctest::Approx(0.0));

    // A flattened strip spans a real extent in both directions.
    double umin = 0, umax = 0, vmin = 0, vmax = 0;
    cv::minMaxLoc(u, &umin, &umax);
    cv::minMaxLoc(v, &vmin, &vmax);
    CHECK(umax - umin > 100.0);
    CHECK(vmax - vmin > 100.0);

    fs::remove_all(dir.parent_path());
}
