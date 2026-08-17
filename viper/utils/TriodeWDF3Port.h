#pragma once

#include "QuadricTube.h"

class TriodeWDF3Port {
public:
    TriodeWDF3Port();

    void SetTubeModel(const TubeModel& model);
    void SetPortResistances(double z1, double z2, double z3);

    void SetIncident(double a1, double a2, double a3) {
        a1_ = a1;
        a2_ = a2;
        a3_ = a3;
    }

    void Scatter();

    double B1() const { return b1_; }
    double B2() const { return b2_; }
    double B3() const { return b3_; }

    double Vpk() const {
        return 0.5 * (a1_ + b1_) - 0.5 * (a3_ + b3_);
    }

    double Vgk() const {
        return 0.5 * (a2_ + b2_) - 0.5 * (a3_ + b3_);
    }

    double Ip() const {
        return (a1_ - b1_) * inv_2z1_;
    }

private:
    void UpdateCoefficients();

    TubeModel model_{};
    double z1_ = 1.0;
    double z2_ = 1.0;
    double z3_ = 0.0;

    double a1_ = 0.0;
    double a2_ = 0.0;
    double a3_ = 0.0;

    double b1_ = 0.0;
    double b2_ = 0.0;
    double b3_ = 0.0;

    // Precomputed coefficients for Scatter() and Ip()
    bool is_valid_ = false;
    double z3_over_z1_ = 0.0;
    double half_z3_over_z1_ = 0.0;
    double beta_a1_coeff_ = 0.5;
    double inv_kp2_gamma_ = 0.0;
    double inv_term8_ = 0.0;
    double inv_term4_ = 0.0;
    double inv_sqrt_2z1kp2_gamma_ = 0.0;
    double vpk_neg_b1_a1_ = -1.0;
    double vpk_neg_b1_a3_ = 2.0;
    double inv_2z1_ = 0.5;
};