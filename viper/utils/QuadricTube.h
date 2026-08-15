#pragma once

#include <cstdint>

// Struct to hold the 3 Quadric parameters for any tube
struct TubeModel {
    double kp;
    double kp2;
    double kpg;
};

class QuadricTube {
public:
    QuadricTube();

    double Process(double sample);
    void Reset();

    // Use this to change the tube model dynamically
    void SetTubeModel(const TubeModel& model, double vdd, double rp, double bias);

    void SetDrive(double drive);

private:
    double vdd_           = 0.0;
    double rp_            = 1.0;   // Non-zero to avoid division by zero in k_A_
    double bias_          = 0.0;
    double output_scale_  = 0.0;
    double drive_factor_  = 2.0;   // Input gain multiplier [1.0 – 10.0]; default 2.0
    TubeModel tube_       = {};

    double k_A_       = 0.0, k_2A_      = 0.0, k_4A_      = 0.0;
    double k_B_const_ = 0.0, k_B_vgk_  = 0.0;
    double k_C_vgk2_  = 0.0, k_C_vgk_  = 0.0, k_C_const_ = 0.0;

    double last_processed_ = 0.0;
    double prev_out_       = 0.0;
};
