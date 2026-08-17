#include "TriodeWDF3Port.h"

#include <cmath>

namespace {
constexpr double kEps = 1e-12;

inline void SetIdentityScatter(double& b1, double& b2, double& b3, double a1, double a2, double a3) {
    b1 = a1;
    b2 = a2;
    b3 = a3;
}

inline double ComputeB3(double a1, double a3, double b1, double z3_over_z1) {
    return a3 + z3_over_z1 * (a1 - b1);
}

inline bool IsInvalidModel(double kp2, double kp, double kpg, double z1, double z3) {
    return !std::isfinite(kp2) || !std::isfinite(kp) || !std::isfinite(kpg)
        || !std::isfinite(z1) || !std::isfinite(z3)
        || kp2 <= 0.0 || z1 <= 0.0;
}

inline bool IsInvalidScatterTerm(double value) {
    return !std::isfinite(value) || value < kEps;
}
}

TriodeWDF3Port::TriodeWDF3Port() {
    UpdateCoefficients();
}

void TriodeWDF3Port::SetTubeModel(const TubeModel& model) {
    model_ = model;
    UpdateCoefficients();
}

void TriodeWDF3Port::SetPortResistances(double z1, double z2, double z3) {
    z1_ = (std::isfinite(z1) && z1 > 0.0) ? z1 : 1.0;
    z2_ = std::isfinite(z2) ? z2 : 1.0;
    z3_ = std::isfinite(z3) ? z3 : 0.0;
    UpdateCoefficients();
}

void TriodeWDF3Port::UpdateCoefficients() {
    const double kp2 = model_.kp2;
    const double kp = model_.kp;
    const double kpg = model_.kpg;

    if (IsInvalidModel(kp2, kp, kpg, z1_, z3_)) {
        is_valid_ = false;
        return;
    }

    const double z3_over_z1 = z3_ / z1_;
    const double gamma = 1.0 + z3_over_z1 * (1.0 + kpg / (4.0 * kp2));
    if (!std::isfinite(gamma) || std::abs(gamma) < kEps) {
        is_valid_ = false;
        return;
    }

    const double gamma2 = gamma * gamma;
    const double term8 = 8.0 * z1_ * kp2 * gamma2;
    const double term4 = 0.5 * term8;
    if (IsInvalidScatterTerm(term8) || IsInvalidScatterTerm(term4)) {
        is_valid_ = false;
        return;
    }

    const double sqrt_2z1kp2 = std::sqrt(2.0 * z1_ * kp2);
    if (!std::isfinite(sqrt_2z1kp2) || sqrt_2z1kp2 < kEps) {
        is_valid_ = false;
        return;
    }

    const double z1_plus_z3 = z1_ + z3_;
    if (!std::isfinite(z1_plus_z3) || std::abs(z1_plus_z3) < kEps) {
        is_valid_ = false;
        return;
    }

    const double denom_sqrt_gamma = sqrt_2z1kp2 * gamma;
    const double denom_kp2_gamma = kp2 * gamma;
    if (!std::isfinite(denom_sqrt_gamma) || std::abs(denom_sqrt_gamma) < kEps
        || !std::isfinite(denom_kp2_gamma) || std::abs(denom_kp2_gamma) < kEps) {
        is_valid_ = false;
        return;
    }

    z3_over_z1_ = z3_over_z1;
    half_z3_over_z1_ = z3_ / (2.0 * z1_);
    beta_a1_coeff_ = 0.5 * (1.0 - z3_over_z1);
    inv_kp2_gamma_ = 1.0 / denom_kp2_gamma;
    inv_term8_ = 1.0 / term8;
    inv_term4_ = 1.0 / term4;
    inv_sqrt_2z1kp2_gamma_ = 1.0 / denom_sqrt_gamma;
    vpk_neg_b1_a1_ = (z3_ - z1_) / z1_plus_z3;
    vpk_neg_b1_a3_ = (2.0 * z1_) / z1_plus_z3;
    inv_2z1_ = 1.0 / (2.0 * z1_);

    is_valid_ = true;
}

void TriodeWDF3Port::Scatter() {
    b2_ = a2_; // grid current is zero in this WDF triode model

    if (!is_valid_) {
        SetIdentityScatter(b1_, b2_, b3_, a1_, a2_, a3_);
        return;
    }

    const double kp2 = model_.kp2;
    const double kp = model_.kp;
    const double kpg = model_.kpg;

    const double alpha = kp + kpg * (a2_ - a3_ - half_z3_over_z1_ * a1_);
    const double beta = kp2 * (beta_a1_coeff_ * a1_ - a3_);
    const double eta = (beta + 0.5 * alpha) * inv_kp2_gamma_;

    if (!std::isfinite(alpha) || !std::isfinite(beta) || !std::isfinite(eta)) {
        SetIdentityScatter(b1_, b2_, b3_, a1_, a2_, a3_);
        return;
    }

    if (double delta = inv_term8_ + a1_ + eta;
            std::isfinite(delta) && delta >= -1e-15) {
        if (delta < 0.0) delta = 0.0;

        const double sqrt_delta = std::sqrt(delta);
        b1_ = (sqrt_delta * inv_sqrt_2z1kp2_gamma_)
            - inv_term4_
            - eta;

        if (b1_ < -eta) {
            b1_ = a1_;
            b3_ = a3_;
        } else {
            b3_ = ComputeB3(a1_, a3_, b1_, z3_over_z1_);
        }
    } else {
        b1_ = a1_;
        b3_ = a3_;
    }

    if (!std::isfinite(b1_) || !std::isfinite(b3_)) {
        SetIdentityScatter(b1_, b2_, b3_, a1_, a2_, a3_);
        return;
    }

    const double vpk = 0.5 * (a1_ + b1_) - 0.5 * (a3_ + b3_);
    if (vpk >= 0.0) return;

    b1_ = vpk_neg_b1_a1_ * a1_ + vpk_neg_b1_a3_ * a3_;
    b3_ = ComputeB3(a1_, a3_, b1_, z3_over_z1_);

    if (!std::isfinite(b1_) || !std::isfinite(b3_)) {
        SetIdentityScatter(b1_, b2_, b3_, a1_, a2_, a3_);
    }
}