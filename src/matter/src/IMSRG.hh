#ifndef IMSRG_h
#define IMSRG_h 1
#include <deque>
#include "Operator.hh"
#include "Commutator.hh"
class IMSRG{
  public:
    ModelSpace* modelspace;
    Operator* Hin;
    Operator Eta;
    Operator Hs;
    Operator H_saved;
    std::deque<Operator> Omega_s;
    int istep;
    double s;
    double ds;
    double ds_max;
    double smax;
    double norm_domega;
    double omega_norm_max;
    double eta_criterion;
    const double bch_transform_threshold = 1e-9;
    const double bch_product_threshold = 1e-4;

    ~IMSRG(){};
    IMSRG(double smax, double ds_max, double omega_norm_max, double eta_criterion);
    IMSRG(Operator& H_in, double smax, double ds_max, double omega_norm_max, double eta_criterion);
    void Solve(std::string method);
    Operator BCHTransformation(const Operator& op, const Operator& Omega);
    Operator Evolve(const Operator& op);
    Operator BCHProduct(const Operator& op, const Operator& Omega);
    void WriteFlowStatusHeader(std::ostream &f);
    void WriteFlowStatus(std::ostream &f);

    void solve_magnus();
    void solve_runge_kutta4();
    void NewOmega();
};
#endif
