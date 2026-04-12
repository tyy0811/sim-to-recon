#pragma once

#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <optional>

namespace simtorecon {

struct CalibrationResult {
    cv::Mat K;                    // 3x3 intrinsics matrix
    cv::Mat distortion;           // distortion coefficients (5 or 8)
    double reprojection_error;    // RMS reprojection error in pixels
    cv::Size image_size;          // (width, height) of calibration images
};

/// Compute 3D object points for an idealized chessboard pattern.
/// Returns n_rows * n_cols points at z=0, spaced by square_size_mm.
std::vector<cv::Point3f> compute_object_points(
    cv::Size pattern_size,
    float square_size_mm
);

/// Detect chessboard corners in a grayscale or BGR image.
/// Returns empty vector if detection fails.
std::vector<cv::Point2f> detect_chessboard(
    const cv::Mat& img,
    cv::Size pattern_size
);

/// Run camera calibration from multiple views.
/// Requires at least 3 views; returns nullopt if calibration fails.
std::optional<CalibrationResult> calibrate(
    const std::vector<std::vector<cv::Point2f>>& image_points,
    const std::vector<std::vector<cv::Point3f>>& object_points,
    cv::Size image_size
);

/// Serialize a CalibrationResult to JSON.
nlohmann::json to_json(const CalibrationResult& result);

/// Deserialize a CalibrationResult from JSON.
CalibrationResult from_json(const nlohmann::json& j);

}  // namespace simtorecon
