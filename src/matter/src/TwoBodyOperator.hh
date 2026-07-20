#ifndef TwoBodyOperator_h
#define TwoBodyOperator_h 1
#include <iostream>
#include <iomanip>
#include <vector>
#include <complex>
#include <map>
#include <Eigen/Dense>
#include "ModelSpace.hh"
#include "Profiler.hh"
#include "NNPWtoGrid.hh"

class TwoBodyOperator
{
  public:
    ModelSpace * modelspace;
    int rankTz = 0; // this should be 0 for Hamiltonian, but 1 for beta decay operators
    int Qx = 0;  // x component of momentum transfer |Nx' - Nx|
    int Qy = 0;  // y component of momentum transfer |Ny' - Ny|
    int Qz = 0;  // z component of momentum transfer |Nz' - Nz|
    std::map<std::array<int,2>, Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>> Matrices;
    mutable std::map<std::array<int,2>, Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>> XMatrices;
    bool AntiSym = false;

    mutable bool XAllocated = false;
    mutable bool XValid     = false;

    ~TwoBodyOperator();
    TwoBodyOperator();
    TwoBodyOperator(ModelSpace&, int rankTz, int Qx, int Qy, int Qz);
    TwoBodyOperator& operator*=(const double);
    TwoBodyOperator& operator+=(const TwoBodyOperator&);
    TwoBodyOperator& operator-=(const TwoBodyOperator&);

    void SetAntiSym(bool asym) {AntiSym = asym;};
    void SetME(int, int, int, int, std::complex<double>);
    void PrintOperator(std::string) const;
    std::complex<double> GetME(int, int, int, int) const;
    std::complex<double> GetXME(int, int, int, int) const;
    void SetME(SPState& p, SPState& q, SPState& r, SPState& s, std::complex<double> me) {SetME(p.index, q.index, r.index, s.index, me);};
    std::complex<double> GetME(SPState& p, SPState& q, SPState& r, SPState& s) const {return GetME(p.index, q.index, r.index, s.index);};
    void SetNNInteraction(int ch_order, std::map<std::string,double> LECs, double RegLambda, int reg_power, double SFRLambda, bool chiral_delta);
    void SetNNInteraction(std::string);
    void SetNO2B3NInteraction(int ch_order, std::map<std::string,double> LECs, double RegLambda, int reg_power, double SFRLambda, bool chiral_delta);
    void AllocateXOperator() const;
    void SetXOperator() const;
    void UpdateFromXOperator();
    void EraseXOperator() const;
    double Norm() const;
    void Erase();
    void Clear();
    double Memory() const;
    double XMemory() const;

    void EnsureXOperator() const;
    void InvalidateXOperator();
};
#endif

