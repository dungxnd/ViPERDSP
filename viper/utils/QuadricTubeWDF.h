#pragma once

#include "QuadricTube.h"
#include "TriodeWDF3Port.h"

class QuadricTubeWDF {
public:
    QuadricTubeWDF() = default;

    double Process(double sample);
    void Reset();

    void SetTubeModel(const TubeModel& model, double vdd, double rp, double bias,
                      double output_gain = 1.0);
    void SetDrive(double drive);

private:
    bool configured_ = false;
    double vdd_ = 0.0;
    double rp_ = 1.0;
    double bias_ = 0.0;
    double output_scale_ = 0.0;
    double output_gain_ = 1.0;
    double drive_factor_ = 2.0;
    TriodeWDF3Port triode_;
    double last_processed_ = 0.0;
    double prev_out_ = 0.0;
};
