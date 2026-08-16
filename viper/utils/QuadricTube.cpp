#include "QuadricTube.h"
#include <cmath>

QuadricTube::QuadricTube() = default;

void QuadricTube::SetTubeModel(const TubeModel& model, double vdd, double rp, double bias,
                                double output_gain) {
    // Validate inputs — reject configurations that would cause div-by-zero or
    // degenerate arithmetic.  Leave configured_ false so Process()/Reset() no-op.
    if (vdd <= 0.0 || rp <= 0.0 || model.kp2 <= 0.0) {
        configured_ = false;
        return;
    }

    tube_ = model;
    vdd_ = vdd;
    rp_ = rp;
    bias_ = bias;
    output_gain_ = output_gain;

    output_scale_ = -1.0 / (vdd_ / 2.5);

    k_A_  = tube_.kp2 * (rp_ * rp_);
    k_2A_ = 2.0 * k_A_;
    k_4A_ = 4.0 * k_A_;

    k_B_const_ = (-2.0 * tube_.kp2 * vdd_ * rp_) - (tube_.kp * rp_) - 1.0;
    k_B_vgk_   = -(tube_.kpg * rp_);

    k_C_vgk2_  = (tube_.kpg * tube_.kpg) / (4.0 * tube_.kp2);
    k_C_vgk_   = (tube_.kpg * vdd_) + ((tube_.kp * tube_.kpg) / (2.0 * tube_.kp2));
    k_C_const_ = (tube_.kp2 * vdd_ * vdd_) + (tube_.kp * vdd_) + ((tube_.kp * tube_.kp) / (4.0 * tube_.kp2));

    configured_ = true;
    Reset();
}

double QuadricTube::Process(const double sample) {
    if (!configured_) return 0.0;

    const double prev_last = last_processed_;
    const double v_gk = (sample * drive_factor_) + bias_;

    const double B = k_B_const_ + (k_B_vgk_ * v_gk);
    const double C = (k_C_vgk2_ * v_gk * v_gk) + (k_C_vgk_ * v_gk) + k_C_const_;

    const double discriminant = (B * B) - (k_4A_ * C);
    double i_p = 0.0;

    if (discriminant >= -1e-12) {
        const double disc = (discriminant < 0.0) ? 0.0 : discriminant;
        i_p = (-B - std::sqrt(disc)) / k_2A_;
        if (i_p < 0.0) i_p = 0.0;
    }

    double v_pk = vdd_ - (i_p * rp_);
    if (v_pk < 0.0) v_pk = 0.0;
    if (v_pk > vdd_) v_pk = vdd_;

    const double y = v_pk * output_scale_ * output_gain_;
    last_processed_ = y;
    prev_out_ = last_processed_ + prev_out_ * 0.999 - prev_last;

    return prev_out_;
}

void QuadricTube::SetDrive(double drive) {
    if (drive < 1.0)  drive = 1.0;
    if (drive > 10.0) drive = 10.0;
    drive_factor_ = drive;
}

void QuadricTube::Reset() {
    if (!configured_) return;

    const double v_gk = bias_;
    const double B = k_B_const_ + (k_B_vgk_ * v_gk);
    const double C = (k_C_vgk2_ * v_gk * v_gk) + (k_C_vgk_ * v_gk) + k_C_const_;
    const double discriminant = (B * B) - (k_4A_ * C);

    double i_p = 0.0;
    if (discriminant >= -1e-12) {
        const double disc = (discriminant < 0.0) ? 0.0 : discriminant;
        i_p = (-B - std::sqrt(disc)) / k_2A_;
        if (i_p < 0.0) i_p = 0.0;
    }

    double v_pk = vdd_ - (i_p * rp_);
    if (v_pk < 0.0) v_pk = 0.0;
    if (v_pk > vdd_) v_pk = vdd_;

    last_processed_ = v_pk * output_scale_ * output_gain_;
    prev_out_ = 0.0;
}
