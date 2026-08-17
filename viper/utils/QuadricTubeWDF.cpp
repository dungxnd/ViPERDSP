#include "QuadricTubeWDF.h"

#include <cmath>

void QuadricTubeWDF::SetTubeModel(const TubeModel& model, double vdd, double rp, double bias,
                                  double output_gain) {
    if (vdd <= 0.0 || rp <= 0.0 || model.kp2 <= 0.0) {
        configured_ = false;
        return;
    }

    vdd_ = vdd;
    rp_ = rp;
    bias_ = bias;
    output_gain_ = output_gain;
    output_scale_ = -1.0 / (vdd_ / 2.5);

    triode_.SetTubeModel(model);
    triode_.SetPortResistances(rp_, 1.0, 0.0);

    configured_ = true;
    Reset();
}

double QuadricTubeWDF::Process(double sample) {
    if (!configured_) return 0.0;

    const double prev_last = last_processed_;
    const double bias_scale = std::sqrt(std::abs(bias_) / 1.5);
    double v_gk = sample * drive_factor_ * bias_scale + bias_;
    if (v_gk > 0.0) {
        v_gk = 0.5 * std::tanh(v_gk * 2.0);
    }

    triode_.SetIncident(vdd_, v_gk, 0.0);
    triode_.Scatter();

    double v_pk = triode_.Vpk();
    if (v_pk < 0.0) v_pk = 0.0;
    if (v_pk > vdd_) v_pk = vdd_;

    const double y = v_pk * output_scale_ * output_gain_;
    last_processed_ = y;
    prev_out_ = last_processed_ + prev_out_ * 0.999 - prev_last;

    return prev_out_;
}

void QuadricTubeWDF::Reset() {
    if (!configured_) return;

    const double v_gk = bias_;
    triode_.SetIncident(vdd_, v_gk, 0.0);
    triode_.Scatter();

    double v_pk = triode_.Vpk();
    if (v_pk < 0.0) v_pk = 0.0;
    if (v_pk > vdd_) v_pk = vdd_;

    last_processed_ = v_pk * output_scale_ * output_gain_;
    prev_out_ = 0.0;
}

void QuadricTubeWDF::SetDrive(double drive) {
    if (drive < 1.0) drive = 1.0;
    if (drive > 10.0) drive = 10.0;
    drive_factor_ = drive;
}
