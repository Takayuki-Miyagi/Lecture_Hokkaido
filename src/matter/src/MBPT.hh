#ifndef MBPT_h
#define MBPT_h 1
#include <iostream>
#include <iomanip>
#include <vector>
#include <complex>
#include <map>
#include <Eigen/Dense>
#include "ModelSpace.hh"
#include "Operator.hh"

class MBPT
{
  public:
    ~MBPT(){};
    MBPT(){};
    std::complex<double> EnergyCorrection2(const Operator& Ham);
};
#endif
