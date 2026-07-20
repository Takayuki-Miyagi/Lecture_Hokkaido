#ifndef OneBodyOperator_h
#define OneBodyOperator_h 1
#include <iostream>
#include <iomanip>
#include <vector>
#include <complex>
#include <map>
#include <Eigen/Dense>
#include "ModelSpace.hh"

class OneBodyOperator
{
  public:
    ModelSpace * modelspace;
    int rankTz = 0;
    int Qx = 0;  // x component of momentum transfer |Nx' - Nx|
    int Qy = 0;  // y component of momentum transfer |Ny' - Ny|
    int Qz = 0;  // z component of momentum transfer |Nz' - Nz|
    Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> Matrix; // <p|O|q>
    bool AntiSym = false;
    std::map<int,std::vector<int>> non_zero_loop_indices;

    ~OneBodyOperator();
    OneBodyOperator();
    OneBodyOperator(ModelSpace&, int, int, int, int);
    OneBodyOperator& operator*=(const double);
    OneBodyOperator& operator+=(const OneBodyOperator&);
    OneBodyOperator& operator-=(const OneBodyOperator&);

    void SetAntiSym(bool asym) {AntiSym = asym;};
    void SetME(int p, int q, std::complex<double> me){Matrix(p,q) = me;};
    std::complex<double> GetME(int p, int q) const {return Matrix(p,q);};
    void SetME(SPState& p, SPState& q, std::complex<double> me){Matrix(p.index,q.index) = me;};
    std::complex<double> GetME(SPState& p, SPState& q) const {return Matrix(p.index,q.index);};
    void SetKineticEnergy();
    void PrintOperator() {std::cout << Matrix << std::endl;};
    void SetNonZeroLoopIndices();
    void SetSpinOperator();
    std::vector<int> GetNonZeroLoop(int ip) const {return non_zero_loop_indices.at(ip);};
    double Norm() const;
    void Erase();
    double Memory();
    bool IsZero(double threshold) const;
    bool IsDiagonal(double threashold) const;
};
#endif
