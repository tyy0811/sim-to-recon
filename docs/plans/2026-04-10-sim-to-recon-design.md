# sim-to-recon — Implementation Plan

**Multi-view 3D reconstruction benchmark with stress-tested honest evaluation, applied to posed-image dense MVS as the geometric sibling of [sim-to-data](https://github.com/tyy0811/sim-to-data).**

Status: V1 build started 2026-04-10. See the full spec in the original conversation.

## V1 Scope (5-7 days)

- One DTU scene (scan9, 49 images), COLMAP dense MVS via pycolmap
- C++17/CMake/OpenCV calibration binary with 8 GoogleTest tests
- Evaluation: chamfer, accuracy, completeness, F-score vs DTU GT
- Stress: view-count sweep at N in {49, 30, 15, 8}
- ~20 tests (pytest + GoogleTest), CI with lint + tests + C++ build
- README in sim-to-data structure, DECISIONS.md with 10 entries

## V1.5 Scope (conditional, 5-7 days after V1)

- 3DGS comparison via gsplat on Modal GPU
- Phone-shot scenes with self-calibration
- Three additional stress dimensions (calib noise, blur, texture)
- MLflow tracking, multi-seed for 3DGS

## V2 Scope (highly conditional, 5-7 days after V1.5)

- Selective reconstruction with conformal calibration (lifted from sim-to-data)
- Cost-sensitive deployment analysis
- Math appendix on conformal prediction for 3D

## Day-by-Day Plan (V1)

1. Setup, data, skeleton
2. COLMAP pipeline end-to-end
3. C++ calibration module
4. Evaluation harness
5. View count stress sweep
6. README, DECISIONS, polish
7. Buffer
