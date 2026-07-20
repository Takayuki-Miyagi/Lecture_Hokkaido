#ifndef NNPWtoGrid_hh
#define NNPWtoGrid_hh 1
#include <iostream>
#include <iomanip>
#include <vector>
#include <complex>
#include <map>
#include <cmath>
#include <Eigen/Dense>
#include "Constants.hh"
#include "ModelSpace.hh"
#include "NNPartialWave.hh"

class NNPWtoGrid
{
  private:
    struct MomentumGrid {
      public:
        int ix, iy, iz;       // index of x, y, z ( Cartesian)
        double px, py, pz;    // p = (px, py, pz )   Cartesian
        double p, theta, phi; // p = (p, theta, phi) Spherical

        MomentumGrid(int ix_, int iy_, int iz_,
            double px_, double py_, double pz_,
            double p_, double theta_, double phi_)
          : ix(ix_), iy(iy_), iz(iz_),
          px(px_), py(py_), pz(pz_),
          p(p_), theta(theta_), phi(phi_) {};
    };

    class MomentumGrids {
      public:
        std::vector<MomentumGrid> grids;
        std::map<std::array<int,3>,int> grid_to_idx;
        std::map<int,double> abscissa;
        double Lbox = -1.0;
        int n_abscissa, Nmax;
        inline int sign(double x) { return (x > 0) - (x < 0); };

        ~MomentumGrids(){};
        MomentumGrids(double, int);
        int GetNumberGrids() const {return grids.size();};
        int GetGridIndex(int ix, int iy, int iz) const {return grid_to_idx.at({ix,iy,iz});};
    };

    class SpinPNChannel {
      public:
        int sz1, sz2, Tz;
        MomentumGrids * mom;
        ~SpinPNChannel(){};
        SpinPNChannel(int s1, int s2, int t, MomentumGrids& g)
          : sz1(s1), sz2(s2), Tz(t), mom(&g) {};
    };

    class SpinPNChannels {
      public:
        std::vector<SpinPNChannel> channels;
        std::map<std::array<int,3>,int> spin_pn_idx;
        MomentumGrids mom;

        ~SpinPNChannels(){};
        SpinPNChannels(double, int);
        int GetNumberChannels() const {return channels.size();};
        int GetChannelIndex(int sz1, int sz2, int Tz) const {return spin_pn_idx.at({sz1, sz2, Tz});};
        SpinPNChannel& GetChannel(int idx) {return channels[idx];};
    };

    class NNCartesianChannel {
      public:
        Eigen::MatrixXcd matrix;
        SpinPNChannel * channel_bra;
        SpinPNChannel * channel_ket;
        bool is_zero = false;

        ~NNCartesianChannel(){};
        NNCartesianChannel(SpinPNChannel&, SpinPNChannel&);
    };

    void StoreArrays(MomentumGrids&, int);

    class store_spin_cg {
      public:
        int size_m1, size_m2, size_M, size_S, total_size;
        std::vector<double> arr;

        ~store_spin_cg(){};
        store_spin_cg();
        int cache(int m1, int m2, int S, int Ms);
        double at(int m1, int m2, int S, int Ms) {return arr[cache(m1, m2, S, Ms)];};
    };

    class store_lsj_cg {
      public:
        int Jmax;
        int size_l, size_ml, size_s, size_ms, size_j, size_m, total_size;
        std::vector<double> arr;
        ~store_lsj_cg(){};
        store_lsj_cg(int);
        int cache(int l, int ml, int s, int ms, int j, int m);
        double at(int l, int ml, int s, int ms, int j, int m) {return arr[cache(l,ml,s,ms,j,m)];};
    };

    class store_Ylm {
      public:
        int Lmax;
        int size_idx, size_l, size_m, total_size;
        std::vector<std::complex<double>> arr;
        ~store_Ylm(){};
        store_Ylm(MomentumGrids&, int);
        int cache(int idx, int l, int m);
        std::complex<double> at(int idx, int l, int m) {return arr[cache(idx, l, m)];};
    };

  public:
    double Lbox;
    int Nmax;
    std::map<std::array<int,2>,NNCartesianChannel> Matrices;
    SpinPNChannels spin_pn_channels;

    ~NNPWtoGrid(){};
    NNPWtoGrid(double, int);

    void SetNNInteraction(std::string);
    void FindInterpolants(Eigen::VectorXd&, Eigen::VectorXi&, NNPartialWave&);
    void Interpolate(NNPartialWave&, NNPartialWave&);
    void ToGridBasis(NNPartialWave&, Eigen::VectorXi&, store_spin_cg&, store_lsj_cg&, store_Ylm&);
    std::complex<double> GetME(std::array<int,5>, std::array<int,5>, std::array<int,5>, std::array<int,5>);
    std::complex<double> GetME(int, int, int, int, int, int, int, int, int, int, int, int);
};

#endif
