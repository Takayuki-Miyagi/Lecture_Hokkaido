#ifndef Operator_h
#define Operator_h 1
#include <iostream>
#include <iomanip>
#include <vector>
#include <complex>
#include <map>
#include <Eigen/Dense>
#include "ModelSpace.hh"
#include "OneBodyOperator.hh"
#include "TwoBodyOperator.hh"

class Operator
{
  public:
    ModelSpace * modelspace;
    int rankTz = 0;
    int Qx = 0;
    int Qy = 0;
    int Qz = 0;
    std::complex<double> ZeroBody;
    OneBodyOperator OneBody;
    TwoBodyOperator TwoBody;
    TwoBodyOperator V3NO2B;
    bool normal_ordered = false;
    bool antisym = false;

    ~Operator();
    Operator() { Profiler::counter["N_Operators"]++; };
    Operator(ModelSpace&, std::optional<int> rankTz=0, std::optional<int> Qx=0, std::optional<int> Qy=0, std::optional<int> Qz=0);
    Operator(const Operator&); // copy
    Operator(Operator&&);
    // operator overload
    Operator& operator=( const Operator&);
    Operator& operator=( Operator&&);

    Operator& operator+=( const Operator&);
    Operator operator+( const Operator&) const;
    Operator& operator+=( const double&);
    Operator operator+( const double&) const;
    Operator& operator-=( const Operator&);
    Operator operator-( const Operator&) const;
    Operator operator-( ) const;
    Operator& operator-=( const double&);
    Operator operator-( const double&) const;
    Operator& operator*=( const double);
    Operator operator*( const double) const;
    Operator& operator/=( const double);
    Operator operator/( const double) const;


    void Allocate(ModelSpace& ms, int rankTz, int Qx, int Qy, int Qz);
    void AllocateV3NO2B();
    void SetAntiSym(bool asym) {OneBody.SetAntiSym(asym); TwoBody.SetAntiSym(asym); antisym=asym;};
    void PrintEnergy();
    void PrintSPE();
    void PrintOperator(std::string);
    void Set1BME(int ip, int iq, std::complex<double> me) {OneBody.SetME(ip, iq, me);};
    std::complex<double> Get1BME(int ip, int iq) const {return OneBody.GetME(ip, iq);};
    void Set2BME(int ip, int iq, int ir, int is, std::complex<double> me) {TwoBody.SetME(ip, iq, ir, is, me);};
    std::complex<double> Get2BME(int ip, int iq, int ir, int is) const {return TwoBody.GetME(ip, iq, ir, is);};

    void Set1BME(SPState& p, SPState& q, std::complex<double> me) {OneBody.SetME(p, q, me);};
    std::complex<double> Get1BME(SPState& p, SPState& q) const {return OneBody.GetME(p, q);};
    void Set2BME(SPState& p, SPState& q, SPState& r, SPState& s, std::complex<double> me) {TwoBody.SetME(p, q, r, s, me);};
    std::complex<double> Get2BME(SPState& p, SPState& q, SPState& r, SPState& s) const {return TwoBody.GetME(p, q, r, s);};
    void NormalOrderingH();
    void NormalOrdering();
    void SetGenerator(Operator& Hin);
    void SetSpinOperator();
    bool IsHamLikeSymmetry() const {return (rankTz==0 and Qx==0 and Qy==0 and Qz==0) ? true : false;};
    double Norm() const;
    void Erase();
    double Memory();
};
Operator operator*(const double lhs, const Operator& rhs);
Operator operator*(const double lhs, const Operator&& rhs);
#endif
