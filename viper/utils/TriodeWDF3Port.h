#pragma once

#include "QuadricTube.h"

class TriodeWDF3Port {
public:
    void SetTubeModel(const TubeModel& model);
    void SetPortResistances(double z1, double z2, double z3);
    void SetIncident(double a1, double a2, double a3);
    void Scatter();

    double B1() const { return b1_; }
    double B2() const { return b2_; }
    double B3() const { return b3_; }

    double Vpk() const;
    double Vgk() const;
    double Ip() const;

private:
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
};
