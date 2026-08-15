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

private:
    double vdd_;
    double rp_;
    double bias_;
    double output_scale_;
    TubeModel tube_;

    double k_A_, k_2A_, k_4A_;
    double k_B_const_, k_B_vgk_;
    double k_C_vgk2_, k_C_vgk_, k_C_const_;

    double last_processed_;
    double prev_out_;
};
