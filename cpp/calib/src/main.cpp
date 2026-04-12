#include "calib.hpp"
#include <cxxopts.hpp>
#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    cxxopts::Options options("calib",
        "Camera calibration from chessboard images.\n"
        "Part of sim-to-recon: multi-view 3D reconstruction benchmark.");

    options.add_options()
        ("images", "Directory containing chessboard images", cxxopts::value<std::string>())
        ("pattern", "Chessboard inner corners (e.g. 9x6)", cxxopts::value<std::string>()->default_value("9x6"))
        ("square", "Square size in mm", cxxopts::value<float>()->default_value("25.0"))
        ("output", "Output JSON file path", cxxopts::value<std::string>()->default_value("calib.json"))
        ("h,help", "Print usage");

    auto result = options.parse(argc, argv);

    if (result.count("help") || !result.count("images")) {
        std::cout << options.help() << std::endl;
        return result.count("help") ? 0 : 1;
    }

    // Parse pattern size
    auto pattern_str = result["pattern"].as<std::string>();
    int pw, ph;
    if (sscanf(pattern_str.c_str(), "%dx%d", &pw, &ph) != 2) {
        std::cerr << "Error: invalid pattern format '" << pattern_str
                  << "', expected NxM (e.g. 9x6)" << std::endl;
        return 1;
    }
    cv::Size pattern_size(pw, ph);
    float square_size = result["square"].as<float>();
    std::string images_dir = result["images"].as<std::string>();
    std::string output_path = result["output"].as<std::string>();

    std::cout << "Calibrating from: " << images_dir << std::endl;
    std::cout << "Pattern: " << pw << "x" << ph
              << ", square: " << square_size << "mm" << std::endl;

    // Collect image paths
    std::vector<fs::path> image_paths;
    for (const auto& entry : fs::directory_iterator(images_dir)) {
        auto ext = entry.path().extension().string();
        if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp") {
            image_paths.push_back(entry.path());
        }
    }
    std::sort(image_paths.begin(), image_paths.end());

    if (image_paths.empty()) {
        std::cerr << "Error: no images found in " << images_dir << std::endl;
        return 1;
    }
    std::cout << "Found " << image_paths.size() << " images" << std::endl;

    // Detect chessboards
    auto obj_pts = simtorecon::compute_object_points(pattern_size, square_size);
    std::vector<std::vector<cv::Point2f>> all_image_points;
    std::vector<std::vector<cv::Point3f>> all_object_points;
    cv::Size image_size;

    for (const auto& path : image_paths) {
        cv::Mat img = cv::imread(path.string());
        if (img.empty()) {
            std::cerr << "Warning: failed to load " << path << std::endl;
            continue;
        }
        if (image_size.empty()) {
            image_size = cv::Size(img.cols, img.rows);
        }

        auto corners = simtorecon::detect_chessboard(img, pattern_size);
        if (!corners.empty()) {
            all_image_points.push_back(corners);
            all_object_points.push_back(obj_pts);
            std::cout << "  [OK] " << path.filename() << std::endl;
        } else {
            std::cout << "  [--] " << path.filename() << " (no corners)" << std::endl;
        }
    }

    std::cout << "Detected corners in " << all_image_points.size()
              << "/" << image_paths.size() << " images" << std::endl;

    // Calibrate
    auto calib = simtorecon::calibrate(
        all_image_points, all_object_points, image_size
    );

    if (!calib.has_value()) {
        std::cerr << "Error: calibration failed (need >= 3 successful detections)"
                  << std::endl;
        return 1;
    }

    // Write JSON
    auto json = simtorecon::to_json(calib.value());
    std::ofstream ofs(output_path);
    ofs << json.dump(2) << std::endl;

    std::cout << "Calibration complete." << std::endl;
    std::cout << "  fx=" << calib->K.at<double>(0, 0)
              << " fy=" << calib->K.at<double>(1, 1) << std::endl;
    std::cout << "  cx=" << calib->K.at<double>(0, 2)
              << " cy=" << calib->K.at<double>(1, 2) << std::endl;
    std::cout << "  reprojection error: " << calib->reprojection_error
              << " px" << std::endl;
    std::cout << "Written to: " << output_path << std::endl;

    return 0;
}
