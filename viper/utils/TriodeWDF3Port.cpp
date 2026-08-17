#include "TriodeWDF3Port.h"

#include <cmath>

void TriodeWDF3Port::SetTubeModel(const TubeModel& model) {
    model_ = model;
}

void TriodeWDF3Port::SetPortResistances(double z1, double z2, double z3) {
    z1_ = z1;
    z2_ = z2;
    z3_ = z3;
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

    if (kp2 <= 0.0 || z1_ <= 0.0) {
        b1_ = a1_;
        b2_ = a2_;
        b3_ = a3_;
        return;
    }

    const double gamma = 1.0 + (z3_ / z1_) * (1.0 + kpg / (4.0 * kp2));
    const double alpha = kp + kpg * (a2_ - a3_ - (z3_ / (2.0 * z1_)) * a1_);
    const double beta = kp2 * (((1.0 - z3_ / z1_) / 2.0) * a1_ - a3_);
    const double eta = (beta + alpha / 2.0) / (kp2 * gamma);

    const double gamma2 = gamma * gamma;
    const double term_8z1kp2_gamma2 = 8.0 * z1_ * kp2 * gamma2;
    const double term_4z1kp2_gamma2 = 4.0 * z1_ * kp2 * gamma2;

    double delta = (1.0 / term_8z1kp2_gamma2) + a1_ + eta;
    bool open_circuit = false;

    if (delta >= -1e-15) {
        if (delta < 0.0) delta = 0.0;

        const double sqrt_delta = std::sqrt(delta);
        const double sqrt_2z1kp2 = std::sqrt(2.0 * z1_ * kp2);

        b1_ = (sqrt_delta / (sqrt_2z1kp2 * gamma))
            - (1.0 / term_4z1kp2_gamma2)
            - eta;

        b3_ = a3_ + (z3_ / z1_) * (a1_ - b1_);
        b2_ = a2_;

        const double vpk = Vpk();
        const double vgk = Vgk();
        const double ip_deriv = kpg * vgk + 2.0 * kp2 * vpk + kp;

        if (ip_deriv < 0.0) open_circuit = true;
    } else {
        open_circuit = true;
    }

    if (open_circuit) {
        b1_ = a1_;
        b3_ = a3_ + (z3_ / z1_) * (a1_ - b1_);
        b2_ = a2_;
    }

    if (Vpk() < 0.0) {
        b1_ = ((z3_ - z1_) * a1_ + 2.0 * z1_ * a3_) / (z1_ + z3_);
        b3_ = a3_ + (z3_ / z1_) * (a1_ - b1_);
        b2_ = a2_;
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
