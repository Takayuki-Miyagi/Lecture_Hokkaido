#ifndef SPState_h
#define SPState_h 1
#include <iostream>
#include <iomanip>
#include <complex>
#include <Eigen/Dense>

class SPState
{
  public:
    // |p> = |index> = |nx,ny,nz,sz,tz>
    int nx, ny, nz; // lattice point
    int sz; // 1 or -1 spin up or down
    int tz;   // 1 (neutron) or -1 (proton)
    double occ; // oocupation number
    int index; // index of the SPState

    ~SPState() {}; // destructor
    SPState() : nx(0), ny(0), nz(0), sz(0), tz(0), occ(-1), index(-1) {}; // constructor
    SPState(int nx, int ny, int nz, int sz, int tz, double occ, int index) : nx(nx), ny(ny), nz(nz), sz(sz), tz(tz), occ(occ), index(index) {}; // Constructor
    SPState(const SPState& orb) : nx(orb.nx), ny(orb.ny), nz(orb.nz), sz(orb.sz), tz(orb.tz), occ(orb.occ), index(orb.index) {}; // Constructor
    const Eigen::Vector<double, 3> GetVector() {Eigen::Vector<double, 3> res; res << nx, ny, nz; return res;}; // return 3d vector pointing the lattice point
    bool operator==(const SPState& orb) const {return (nx==orb.nx and ny==orb.ny and nz==orb.nz and sz==orb.sz and tz==orb.tz); }; // you can do if(p == q)
    void SetParticleHole(double occ_in) {occ = occ_in;};
    friend auto operator<<(std::ostream& os, const SPState& orb)
      -> std::ostream& {return os
        << std::setw(7) << "index:" << std::setw(4) << orb.index << ","
          << std::setw(6) << "nx:" << std::setw(4) << orb.nx << ","
          << std::setw(6) << "ny:" << std::setw(4) << orb.ny << ","
          << std::setw(6) << "nz:" << std::setw(4) << orb.nz << ","
          << std::setw(6) << "sz:" << std::setw(4) << orb.sz << ","
          << std::setw(6) << "tz:" << std::setw(4) << orb.tz << ","
          << std::setw(10) << "length:" << std::fixed << std::setprecision(4) << std::setw(8) << std::pow(orb.nx*orb.nx + orb.ny*orb.ny + orb.nz*orb.nz, 0.5) << ","
          << std::setw(10) << ((orb.occ>1.e-8) ? "hole" : "particle");
      }; // you can do std::cout << p << std::endl;
};
#endif
