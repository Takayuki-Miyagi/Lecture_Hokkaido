#ifndef ModelSpace_h
#define ModelSpace_h 1
#include <iostream>
#include <iomanip>
#include <optional>
#include <vector>
#include <map>
#include "Constants.hh"
#include "SPState.hh"
#include "TwoBodySpace.hh"

class ModelSpace
{
  public:
    int Zu, Zd, Nu, Nd, Z, N, A; // proton, neutron, mass numbers
    int Nmax=-1;  // (-Nmax, -Nmax+1, ..., 0, 1, ..., Nmax)
    double Lbox;
    double k_unit; // 2 * pi / Lbox
    int L2Fermi_pu = -1, L2Fermi_pd = -1, L2Fermi_nu = -1, L2Fermi_nd = -1;
    bool verbose =  true;
    std::vector<SPState> SPStates; // List of orbits
    std::vector<int> all_states;
    std::vector<int> hole_states;
    std::vector<int> particle_states;
    std::string truncation;
    Eigen::Vector<double, 3> k_shift; // momentum for twist angles; theta/L
    TwoBodySpace TwoBody;

    ~ModelSpace();
    ModelSpace(int A, int Nmax, double rho, std::string truncation, std::string mode, std::array<double,3> twist_angles, bool verbose);
    ModelSpace(int Zu, int Zd, int Nu, int Nd, int Nmax, double rho, std:: string truncation, std::array<double,3> twist_angles, bool verbose);
    void Initialize(int Zu, int Zd, int Nu, int Nd, int Nmax, double Lbox, std::array<double,3> twist_angles);
    void SetClosedShellConf();
    double GetDensity(double N, double L) {return N / (L*L*L);};
    double GetDensity(double kF) {return kF*kF*kF / 3.0 / std::pow(Constants::PI,2);};
    double GetFermiMom(double density) {return std::pow(3 * std::pow(Constants::PI,2) * density, 1.0/3.0);};
    double GetFermiMom(double N, double L) {return GetFermiMom(GetDensity(N,L));};
    int GetNmax() {return Nmax;};
    int GetNumberSPStates() {return SPStates.size();}
    void AddToSPStates(int, int, int, int, double, int);
    void SetSPStates();
    void PrintSPStates(std::optional<int> imin=std::nullopt, std::optional<int> imax=std::nullopt);
    SPState& GetSPState(int i) {return SPStates[i];};

    const Eigen::Vector<double, 3> GetMomentum(SPState& s) {return k_unit * s.GetVector() + k_shift;};
    const Eigen::Vector<double, 3> GetSPMomentum(SPState& p) {return k_unit * p.GetVector() + k_shift;};
    const Eigen::Vector<double, 3> Get2BRelMomentum(SPState& p, SPState& q) {return 0.5*k_unit * (p.GetVector() - q.GetVector());};
    const Eigen::Vector<double, 3> Get3BJacMomentumQ(SPState& p, SPState& q, SPState& r) {return 2.0/3.0 * k_unit * (r.GetVector() - 0.5 * (p.GetVector() + q.GetVector()));};
};
#endif

