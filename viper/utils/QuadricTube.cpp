#include "QuadricTube.h"
#include <cmath>

// 12AX7 — high-gain dual triode (ECC83). Datasheet basis:
//   Va=250V, Vg=-1.5V, Ia≈0.77mA → mu≈100, gm≈1.27mA/V, rp≈76kΩ
static constexpr TubeModel TUBE_12AX7 = { 1.014e-5, 5.498e-8, 1.076e-5 };

// 6N1J — Soviet medium-gain dual triode (military 6N1P variant). Datasheet basis:
//   Va=100V, Vg=-1V, Ia=7.5mA → mu=35, gm=4.35mA/V, rp=8046Ω
//   Fitted via QuadricTube closed-form: kp2=(gm/(2·mu·√Ia))², kpg=2·kp2·mu,
//   kp solved from f0=√Ia at the operating point.
//   Use: SetTubeModel(TUBE_6N1J, 250.0, 20000.0, -2.0)
static constexpr TubeModel TUBE_6N1J  = { 5.7349e-5, 5.1490e-7, 3.6043e-5 };

QuadricTube::QuadricTube() {
    SetTubeModel(TUBE_12AX7, 250.0, 100000.0, -1.5);
}

void QuadricTube::SetTubeModel(const TubeModel& model, double vdd, double rp, double bias) {
    tube_ = model;
    vdd_ = vdd;
    rp_ = rp;
    bias_ = bias;

    output_scale_ = -1.0 / (vdd_ / 2.5); 
    
    k_A_  = tube_.kp2 * (rp_ * rp_);
    k_2A_ = 2.0 * k_A_;
    k_4A_ = 4.0 * k_A_;

    k_B_const_ = (-2.0 * tube_.kp2 * vdd_ * rp_) - (tube_.kp * rp_) - 1.0;
    k_B_vgk_   = -(tube_.kpg * rp_);

    k_C_vgk2_  = (tube_.kpg * tube_.kpg) / (4.0 * tube_.kp2);
    k_C_vgk_   = (tube_.kpg * vdd_) + ((tube_.kp * tube_.kpg) / (2.0 * tube_.kp2));
    k_C_const_ = (tube_.kp2 * vdd_ * vdd_) + (tube_.kp * vdd_) + ((tube_.kp * tube_.kp) / (4.0 * tube_.kp2));

    Reset();
}

double QuadricTube::Process(const double sample) {
    const double prev_last = last_processed_;
    const double v_gk = (sample * 2.0) + bias_;

    const double B = k_B_const_ + (k_B_vgk_ * v_gk);
    const double C = (k_C_vgk2_ * v_gk * v_gk) + (k_C_vgk_ * v_gk) + k_C_const_;

    const double discriminant = (B * B) - (k_4A_ * C);
    double i_p = 0.0;
    
    if (discriminant >= 0.0) {
        i_p = (-B - std::sqrt(discriminant)) / k_2A_;
        if (i_p < 0.0) i_p = 0.0;
    }

    double v_pk = vdd_ - (i_p * rp_);
    if (v_pk < 0.0) v_pk = 0.0;
    if (v_pk > vdd_) v_pk = vdd_;

    const double y = v_pk * output_scale_;
    last_processed_ = y;
    prev_out_ = last_processed_ + prev_out_ * 0.999 - prev_last;

    return prev_out_;
}

void QuadricTube::Reset() {
    const double v_gk = bias_;
    const double B = k_B_const_ + (k_B_vgk_ * v_gk);
    const double C = (k_C_vgk2_ * v_gk * v_gk) + (k_C_vgk_ * v_gk) + k_C_const_;
    const double discriminant = (B * B) - (k_4A_ * C);
    
    double i_p = 0.0;
    if (discriminant >= 0.0) {
        i_p = (-B - std::sqrt(discriminant)) / k_2A_;
        if (i_p < 0.0) i_p = 0.0;
    }
    
    double v_pk = vdd_ - (i_p * rp_);
    if (v_pk < 0.0) v_pk = 0.0;
    if (v_pk > vdd_) v_pk = vdd_;

    last_processed_ = v_pk * output_scale_;
    prev_out_ = 0.0;
}
