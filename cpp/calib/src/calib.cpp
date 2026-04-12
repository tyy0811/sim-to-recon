#include "calib.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace simtorecon {

std::vector<cv::Point3f> compute_object_points(
    cv::Size pattern_size,
    float square_size_mm
) {
    std::vector<cv::Point3f> points;
    points.reserve(pattern_size.height * pattern_size.width);
    for (int row = 0; row < pattern_size.height; ++row) {
        for (int col = 0; col < pattern_size.width; ++col) {
            points.emplace_back(
                static_cast<float>(col) * square_size_mm,
                static_cast<float>(row) * square_size_mm,
                0.0f
            );
        }
    }
    return points;
}

std::vector<cv::Point2f> detect_chessboard(
    const cv::Mat& img,
    cv::Size pattern_size
) {
    cv::Mat gray;
    if (img.channels() == 3) {
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = img;
    }

    std::vector<cv::Point2f> corners;
    bool found = cv::findChessboardCorners(
        gray, pattern_size, corners,
        cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE
    );

    if (found) {
        cv::cornerSubPix(
            gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
            cv::TermCriteria(
                cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.001
            )
        );
    } else {
        corners.clear();
    }

    return corners;
}

std::optional<CalibrationResult> calibrate(
    const std::vector<std::vector<cv::Point2f>>& image_points,
    const std::vector<std::vector<cv::Point3f>>& object_points,
    cv::Size image_size
) {
    if (image_points.size() < 3) {
        return std::nullopt;
    }
    if (image_points.size() != object_points.size()) {
        return std::nullopt;
    }

    CalibrationResult result;
    result.image_size = image_size;
    result.K = cv::Mat::eye(3, 3, CV_64F);
    result.distortion = cv::Mat::zeros(5, 1, CV_64F);

    std::vector<cv::Mat> rvecs, tvecs;

    result.reprojection_error = cv::calibrateCamera(
        object_points, image_points, image_size,
        result.K, result.distortion,
        rvecs, tvecs
    );

    return result;
}

nlohmann::json to_json(const CalibrationResult& result) {
    nlohmann::json j;

    // Intrinsics as flat array (row-major 3x3)
    std::vector<double> k_values;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            k_values.push_back(result.K.at<double>(r, c));
    j["intrinsics"] = k_values;

    j["fx"] = result.K.at<double>(0, 0);
    j["fy"] = result.K.at<double>(1, 1);
    j["cx"] = result.K.at<double>(0, 2);
    j["cy"] = result.K.at<double>(1, 2);

    // Distortion coefficients
    std::vector<double> dist_values;
    for (int i = 0; i < result.distortion.rows; ++i)
        dist_values.push_back(result.distortion.at<double>(i, 0));
    j["distortion"] = dist_values;

    j["reprojection_error"] = result.reprojection_error;
    j["image_width"] = result.image_size.width;
    j["image_height"] = result.image_size.height;

    return j;
}

CalibrationResult from_json(const nlohmann::json& j) {
    CalibrationResult result;

    auto k_values = j["intrinsics"].get<std::vector<double>>();
    if (k_values.size() != 9) {
        throw std::runtime_error(
            "Invalid intrinsics array: expected 9 elements, got " +
            std::to_string(k_values.size()));
    }
    result.K = cv::Mat(3, 3, CV_64F);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            result.K.at<double>(r, c) = k_values[r * 3 + c];

    auto dist_values = j["distortion"].get<std::vector<double>>();
    result.distortion = cv::Mat(static_cast<int>(dist_values.size()), 1, CV_64F);
    for (int i = 0; i < static_cast<int>(dist_values.size()); ++i)
        result.distortion.at<double>(i, 0) = dist_values[i];

    result.reprojection_error = j["reprojection_error"].get<double>();
    result.image_size = cv::Size(
        j["image_width"].get<int>(),
        j["image_height"].get<int>()
    );

    return result;
}

}  // namespace simtorecon
