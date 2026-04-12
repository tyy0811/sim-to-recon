#include "calib.hpp"
#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <cmath>

using namespace simtorecon;

// ---- Object Points ----

TEST(ObjectPoints, HaveCorrectShape) {
    auto pts = compute_object_points(cv::Size(9, 6), 25.0f);
    ASSERT_EQ(pts.size(), 54u);
    for (const auto& p : pts) {
        EXPECT_FLOAT_EQ(p.z, 0.0f);
    }
    // Check grid spacing
    EXPECT_FLOAT_EQ(pts[1].x - pts[0].x, 25.0f);
    EXPECT_FLOAT_EQ(pts[9].y - pts[0].y, 25.0f);
}

// ---- Detection ----

TEST(Detection, EmptyImageReturnsNoCorners) {
    cv::Mat black = cv::Mat::zeros(480, 640, CV_8UC1);
    auto corners = detect_chessboard(black, cv::Size(9, 6));
    EXPECT_TRUE(corners.empty());
}

TEST(Detection, SyntheticChessboardDetected) {
    // Generate a synthetic chessboard image using known intrinsics
    const int W = 640, H = 480;
    cv::Size pattern(9, 6);

    // Create a white image and draw black squares
    cv::Mat board = cv::Mat::ones(H, W, CV_8UC1) * 255;
    int sq = 30;  // pixel size of each square
    int ox = (W - (pattern.width + 1) * sq) / 2;
    int oy = (H - (pattern.height + 1) * sq) / 2;

    for (int r = 0; r < pattern.height + 1; ++r) {
        for (int c = 0; c < pattern.width + 1; ++c) {
            if ((r + c) % 2 == 0) {
                cv::rectangle(board,
                    cv::Point(ox + c * sq, oy + r * sq),
                    cv::Point(ox + (c + 1) * sq, oy + (r + 1) * sq),
                    cv::Scalar(0), cv::FILLED);
            }
        }
    }

    auto corners = detect_chessboard(board, pattern);
    EXPECT_EQ(corners.size(), 54u);
}

// ---- Calibration ----

TEST(Calibration, RejectsInsufficientViews) {
    // Only 1 view — should return nullopt
    std::vector<cv::Point2f> img_pts(54);
    std::vector<cv::Point3f> obj_pts(54);
    auto result = calibrate(
        {img_pts}, {obj_pts}, cv::Size(640, 480)
    );
    EXPECT_FALSE(result.has_value());
}

TEST(Calibration, SyntheticPerfectDataReturnsLowError) {
    // Generate synthetic calibration data with known K
    const int W = 640, H = 480;
    cv::Size pattern(9, 6);

    cv::Mat K_true = (cv::Mat_<double>(3, 3) <<
        500.0, 0.0, 320.0,
        0.0, 500.0, 240.0,
        0.0, 0.0, 1.0);
    cv::Mat dist_true = cv::Mat::zeros(5, 1, CV_64F);

    auto obj_pts = compute_object_points(pattern, 25.0f);

    std::vector<std::vector<cv::Point2f>> all_img_pts;
    std::vector<std::vector<cv::Point3f>> all_obj_pts;

    // Generate 10 synthetic views
    for (int i = 0; i < 10; ++i) {
        cv::Mat rvec = (cv::Mat_<double>(3, 1) <<
            0.1 * (i - 5), 0.1 * (i % 3 - 1), 0.0);
        cv::Mat tvec = (cv::Mat_<double>(3, 1) <<
            0.0, 0.0, 300.0 + i * 20.0);

        std::vector<cv::Point2f> projected;
        cv::projectPoints(obj_pts, rvec, tvec, K_true, dist_true, projected);

        // Check all points are inside the image
        bool all_inside = true;
        for (const auto& p : projected) {
            if (p.x < 0 || p.x >= W || p.y < 0 || p.y >= H) {
                all_inside = false;
                break;
            }
        }
        if (!all_inside) continue;

        all_img_pts.push_back(projected);
        all_obj_pts.push_back(obj_pts);
    }

    ASSERT_GE(all_img_pts.size(), 3u);

    auto result = calibrate(all_img_pts, all_obj_pts, cv::Size(W, H));
    ASSERT_TRUE(result.has_value());

    // Reprojection error should be very low on perfect synthetic data
    EXPECT_LT(result->reprojection_error, 0.5);

    // Recovered focal length should be close to ground truth
    double fx = result->K.at<double>(0, 0);
    double fy = result->K.at<double>(1, 1);
    EXPECT_NEAR(fx, 500.0, 50.0);  // within 10%
    EXPECT_NEAR(fy, 500.0, 50.0);
}

// ---- Intrinsics validity ----

TEST(Intrinsics, MatrixIsValid) {
    // Use the same synthetic setup as above
    const int W = 640, H = 480;
    cv::Size pattern(9, 6);
    cv::Mat K_true = (cv::Mat_<double>(3, 3) <<
        500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0);
    cv::Mat dist_true = cv::Mat::zeros(5, 1, CV_64F);
    auto obj_pts = compute_object_points(pattern, 25.0f);

    std::vector<std::vector<cv::Point2f>> all_img_pts;
    std::vector<std::vector<cv::Point3f>> all_obj_pts;

    for (int i = 0; i < 5; ++i) {
        cv::Mat rvec = (cv::Mat_<double>(3, 1) << 0.05 * i, 0.03 * i, 0.0);
        cv::Mat tvec = (cv::Mat_<double>(3, 1) << 0.0, 0.0, 350.0 + i * 30.0);
        std::vector<cv::Point2f> projected;
        cv::projectPoints(obj_pts, rvec, tvec, K_true, dist_true, projected);
        all_img_pts.push_back(projected);
        all_obj_pts.push_back(obj_pts);
    }

    auto result = calibrate(all_img_pts, all_obj_pts, cv::Size(W, H));
    ASSERT_TRUE(result.has_value());

    // K should be 3x3
    EXPECT_EQ(result->K.rows, 3);
    EXPECT_EQ(result->K.cols, 3);

    // fx, fy > 0
    EXPECT_GT(result->K.at<double>(0, 0), 0.0);
    EXPECT_GT(result->K.at<double>(1, 1), 0.0);

    // Principal point inside image bounds
    double cx = result->K.at<double>(0, 2);
    double cy = result->K.at<double>(1, 2);
    EXPECT_GT(cx, 0.0);
    EXPECT_LT(cx, W);
    EXPECT_GT(cy, 0.0);
    EXPECT_LT(cy, H);
}

// ---- Distortion ----

TEST(Distortion, CoefficientsHaveCorrectShape) {
    const int W = 640, H = 480;
    cv::Size pattern(9, 6);
    cv::Mat K_true = (cv::Mat_<double>(3, 3) <<
        500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0);
    cv::Mat dist_true = cv::Mat::zeros(5, 1, CV_64F);
    auto obj_pts = compute_object_points(pattern, 25.0f);

    std::vector<std::vector<cv::Point2f>> all_img_pts;
    std::vector<std::vector<cv::Point3f>> all_obj_pts;

    for (int i = 0; i < 5; ++i) {
        cv::Mat rvec = (cv::Mat_<double>(3, 1) << 0.05 * i, 0.03 * i, 0.0);
        cv::Mat tvec = (cv::Mat_<double>(3, 1) << 0.0, 0.0, 350.0 + i * 30.0);
        std::vector<cv::Point2f> projected;
        cv::projectPoints(obj_pts, rvec, tvec, K_true, dist_true, projected);
        all_img_pts.push_back(projected);
        all_obj_pts.push_back(obj_pts);
    }

    auto result = calibrate(all_img_pts, all_obj_pts, cv::Size(W, H));
    ASSERT_TRUE(result.has_value());

    // Should have 5 distortion coefficients (standard model)
    EXPECT_EQ(result->distortion.rows, 5);
    EXPECT_EQ(result->distortion.cols, 1);
}

// ---- JSON round-trip ----

TEST(Json, RoundTrip) {
    CalibrationResult original;
    original.K = (cv::Mat_<double>(3, 3) <<
        500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0);
    original.distortion = (cv::Mat_<double>(5, 1) <<
        0.1, -0.25, 0.001, -0.002, 0.05);
    original.reprojection_error = 0.42;
    original.image_size = cv::Size(640, 480);

    auto j = to_json(original);
    auto recovered = from_json(j);

    // Verify K matches
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            EXPECT_DOUBLE_EQ(
                original.K.at<double>(r, c),
                recovered.K.at<double>(r, c)
            );

    // Verify distortion matches
    for (int i = 0; i < 5; ++i)
        EXPECT_DOUBLE_EQ(
            original.distortion.at<double>(i, 0),
            recovered.distortion.at<double>(i, 0)
        );

    EXPECT_DOUBLE_EQ(original.reprojection_error, recovered.reprojection_error);
    EXPECT_EQ(original.image_size, recovered.image_size);
}
