#include "TriodeWDF3Port.h"

#include <cmath>

namespace {
constexpr double kEps = 1e-12;

void SetIdentityScatter(double& b1, double& b2, double& b3, double a1, double a2, double a3) {
    b1 = a1;
    b2 = a2;
    b3 = a3;
}

bool IsInvalidModel(double kp2, double kp, double kpg, double z1, double z3) {
    return !std::isfinite(kp2) || !std::isfinite(kp) || !std::isfinite(kpg)
        || !std::isfinite(z1) || !std::isfinite(z3)
        || kp2 <= 0.0 || z1 <= 0.0;
}

bool IsInvalidScatterTerm(double value) {
    return !std::isfinite(value) || value < kEps;
}
}

void TriodeWDF3Port::SetTubeModel(const TubeModel& model) {
    model_ = model;
}

void TriodeWDF3Port::SetPortResistances(double z1, double z2, double z3) {
    z1_ = (std::isfinite(z1) && z1 > 0.0) ? z1 : 1.0;
    z2_ = std::isfinite(z2) ? z2 : 1.0;
    z3_ = std::isfinite(z3) ? z3 : 0.0;
}

void TriodeWDF3Port::SetIncident(double a1, double a2, double a3) {
    a1_ = a1;
    a2_ = a2;
    a3_ = a3;
}

void TriodeWDF3Port::Scatter() {
    const double kp2 = model_.kp2;
    const double kp = model_.kp;
    const double kpg = model_.kpg;

    if (IsInvalidModel(kp2, kp, kpg, z1_, z3_)) {
        SetIdentityScatter(b1_, b2_, b3_, a1_, a2_, a3_);
        return;
    }

    const double z3_over_z1 = z3_ / z1_;
    const double gamma = 1.0 + z3_over_z1 * (1.0 + kpg / (4.0 * kp2));
    if (!std::isfinite(gamma) || std::abs(gamma) < kEps) {
        SetIdentityScatter(b1_, b2_, b3_, a1_, a2_, a3_);
        return;
    }

    const double alpha = kp + kpg * (a2_ - a3_ - (z3_ / (2.0 * z1_)) * a1_);
    const double beta = kp2 * (((1.0 - z3_over_z1) / 2.0) * a1_ - a3_);
    const double eta = (beta + alpha / 2.0) / (kp2 * gamma);
    if (!std::isfinite(alpha) || !std::isfinite(beta) || !std::isfinite(eta)) {
        SetIdentityScatter(b1_, b2_, b3_, a1_, a2_, a3_);
        return;
    }

    const double gamma2 = gamma * gamma;
    const double term_8z1kp2_gamma2 = 8.0 * z1_ * kp2 * gamma2;
    const double term_4z1kp2_gamma2 = 4.0 * z1_ * kp2 * gamma2;
    if (IsInvalidScatterTerm(term_8z1kp2_gamma2) || IsInvalidScatterTerm(term_4z1kp2_gamma2)) {
        SetIdentityScatter(b1_, b2_, b3_, a1_, a2_, a3_);
        return;
    }

    double delta = (1.0 / term_8z1kp2_gamma2) + a1_ + eta;
    const bool has_valid_delta = std::isfinite(delta) && delta >= -1e-15;
    if (has_valid_delta) {
        if (delta < 0.0) delta = 0.0;

        const double sqrt_delta = std::sqrt(delta);
        const double sqrt_2z1kp2 = std::sqrt(2.0 * z1_ * kp2);
        if (!std::isfinite(sqrt_delta) || !std::isfinite(sqrt_2z1kp2) || sqrt_2z1kp2 < kEps) {
            SetIdentityScatter(b1_, b2_, b3_, a1_, a2_, a3_);
            return;
        }

        b1_ = (sqrt_delta / (sqrt_2z1kp2 * gamma))
            - (1.0 / term_4z1kp2_gamma2)
            - eta;
        b3_ = a3_ + z3_over_z1 * (a1_ - b1_);
        b2_ = a2_;

        if (!std::isfinite(b1_) || !std::isfinite(b3_)) {
            SetIdentityScatter(b1_, b2_, b3_, a1_, a2_, a3_);
            return;
        }

        if (b1_ < -eta) {
            b1_ = a1_;
            b3_ = a3_ + z3_over_z1 * (a1_ - b1_);
            b2_ = a2_;
            if (!std::isfinite(b3_)) {
                SetIdentityScatter(b1_, b2_, b3_, a1_, a2_, a3_);
                return;
            }
        }
    } else {
        b1_ = a1_;
        b3_ = a3_ + z3_over_z1 * (a1_ - b1_);
        b2_ = a2_;
        if (!std::isfinite(b3_)) {
            SetIdentityScatter(b1_, b2_, b3_, a1_, a2_, a3_);
            return;
        }
    }

    if (Vpk() >= 0.0) return;

    const double z1_plus_z3 = z1_ + z3_;
    if (!std::isfinite(z1_plus_z3) || std::abs(z1_plus_z3) < kEps) {
        SetIdentityScatter(b1_, b2_, b3_, a1_, a2_, a3_);
        return;
    }

    b1_ = ((z3_ - z1_) * a1_ + 2.0 * z1_ * a3_) / z1_plus_z3;
    b3_ = a3_ + z3_over_z1 * (a1_ - b1_);
    b2_ = a2_;
    if (!std::isfinite(b1_) || !std::isfinite(b3_)) {
        SetIdentityScatter(b1_, b2_, b3_, a1_, a2_, a3_);
    }
}

double TriodeWDF3Port::Vpk() const {
    return 0.5 * (a1_ + b1_) - 0.5 * (a3_ + b3_);
}

double TriodeWDF3Port::Vgk() const {
    return 0.5 * (a2_ + b2_) - 0.5 * (a3_ + b3_);
}

double TriodeWDF3Port::Ip() const {
    return (a1_ - b1_) / (2.0 * z1_);
}
