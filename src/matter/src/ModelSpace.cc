#include "Constants.hh"
#include "ModelSpace.hh"

ModelSpace::~ModelSpace()
{}

ModelSpace::ModelSpace(int A, int Nmax, double rho, std::string truncation, std::string mode, std::array<double,3> twist_angles, bool verbose):
  A(A), Nmax(Nmax), truncation(truncation), verbose(verbose)
{
  if(mode == "PNM" or mode == "pnm"){
    Zu = 0; Zd = 0; Nu = A/2; Nd = A/2;
  }
  if(mode == "SNM" or mode == "snm"){
    Zu = A/4; Zd = A/4; Nu = A/4; Nd = A/4;
  }
  Lbox = std::pow(A / rho, 1.0/3.0);
  Initialize(Zu, Zd, Nu, Nd, Nmax, Lbox, twist_angles);
}

ModelSpace::ModelSpace(int Zu, int Zd, int Nu, int Nd, int Nmax, double rho, std::string truncation, std::array<double,3> twist_angles, bool verbose):
  Zu(Zu), Zd(Zd), Nu(Nu), Nd(Nd), Nmax(Nmax), truncation(truncation), verbose(verbose)
{
  A = Zu + Zd + Nu + Nd;
  Lbox = std::pow(A / rho, 1.0/3.0);
  Initialize(Zu, Zd, Nu, Nd, Nmax, Lbox, twist_angles);
}

void ModelSpace::Initialize(int Zu, int Zd, int Nu, int Nd, int Nmax, double Lbox, std::array<double,3> twist_angles)
{
  if(verbose) std::cout << "# Seting up model space" << std::endl;
  Z = Zu + Zd;
  N = Nu + Nd;
  A = Z + N;
  k_unit = 2.0 * Constants::PI / Lbox;
  k_shift << twist_angles[0]/Lbox, twist_angles[1]/Lbox, twist_angles[2]/Lbox;
  if(verbose){
    std::cout << "# proton number (up):  " << std::setw(5) << Zu
      << ", proton number (down):  " << std::setw(5) << Zd << std::endl;
    std::cout << "# neutron number (up): " << std::setw(5) << Nu
      << ", neutron number (down): " << std::setw(5) << Nd << std::endl;
    std::cout << "# proton number: " << std::setw(5) << Z
      << ", neutron number: " << std::setw(5) << N
      << ", mass number: " << std::setw(5) << A << std::endl;
    std::cout << "# Nmax: " << std::setw(5) << Nmax << std::endl;
    std::cout << "# Box size: " << std::fixed << std::setprecision(4) << std::setw(10) << Lbox << " fm"
      << ", max momentum: " << std::fixed << std::setprecision(4) << std::setw(10) << k_unit * std::pow(3*Nmax*Nmax,0.5) << " fm-1" << std::endl;
    std::cout << "# density (proton up): " << std::fixed << std::setprecision(4) << std::setw(7) << GetDensity((double)Zu, Lbox)
      << " fm-3, density (proton down): " << std::fixed << std::setprecision(4) << std::setw(7) << GetDensity((double)Zd, Lbox) << " fm-3" << std::endl;
    std::cout << "# density (neutron up):" << std::fixed << std::setprecision(4) << std::setw(7) << GetDensity((double)Nu, Lbox)
      << " fm-3, density (neutron down):" << std::fixed << std::setprecision(4) << std::setw(7) << GetDensity((double)Nd, Lbox) << " fm-3" << std::endl;
    std::cout << "# total density:" << std::fixed << std::setprecision(4) << std::setw(7) << GetDensity((double)A, Lbox) << " fm-3" << std::endl;
  }
  SetClosedShellConf();
  SetSPStates();
  if(verbose) {
    std::cout << std::endl;
    std::cout << "# Number of single-particle states: " << std::setw(10) << GetNumberSPStates() << std::endl;
    PrintSPStates(std::max(0,A-10), std::min(GetNumberSPStates(),A+30));
    std::cout << std::endl;
  }

  for(int ip = 0; ip < GetNumberSPStates(); ip++){
    SPState & p = GetSPState(ip);
    all_states.push_back(ip);
    if(p.occ > 1.e-8) {
      hole_states.push_back(ip);
    }
    else {
      particle_states.push_back(ip);
    }
  }
  TwoBody = TwoBodySpace(SPStates);
  if(verbose) std::cout << "# Number of NN states: " << std::setw(10) << TwoBody.GetNumberChannels() << std::endl;

}

void ModelSpace::SetClosedShellConf()
{
  int max_length2;
  if(truncation=="box") max_length2 = 3*Nmax*Nmax;
  if(truncation=="shell") max_length2 = Nmax;
  int tmp;
  for (int sz : {1, -1}) {
    for (int tz : {-1, 1}) {

      if(tz == -1 and sz == 1) tmp = Zu;
      if(tz == -1 and sz ==-1) tmp = Zd;
      if(tz ==  1 and sz == 1) tmp = Nu;
      if(tz ==  1 and sz ==-1) tmp = Nd;
      if(tmp == 0) continue;

      for (int length2 = 0; length2 <= max_length2; length2++){
        for (int nx = -Nmax; nx <= Nmax; nx++){
          for (int ny = -Nmax; ny <= Nmax; ny++){
            for (int nz = -Nmax; nz <= Nmax; nz++){
              if(nx*nx + ny*ny + nz*nz != length2) continue;
              tmp -= 1;
            }
          }
        }
        if(tmp == 0) {
          if(tz == -1 and sz == 1) L2Fermi_pu = length2;
          if(tz == -1 and sz ==-1) L2Fermi_pd = length2;
          if(tz ==  1 and sz == 1) L2Fermi_nu = length2;
          if(tz ==  1 and sz ==-1) L2Fermi_nd = length2;
          break;
        }


        if(tmp < 0) {
          std::cout << "Error: you should put a closed-shell like proton or neutron numbers" << std::endl;
          exit(0);
        }
      }
    }
  }
}

void ModelSpace::SetSPStates()
{
  std::vector<int> tmp;
  if(Z > 0) tmp.push_back(-1);
  if(N > 0) tmp.push_back(1);

  int max_length2;
  if(truncation=="box") max_length2 = 3*Nmax*Nmax;
  if(truncation=="shell") max_length2 = Nmax;
  for (int length2 = 0; length2 <= max_length2; length2++){
    for (int nx = -Nmax; nx <= Nmax; nx++){
      for (int ny = -Nmax; ny <= Nmax; ny++){
        for (int nz = -Nmax; nz <= Nmax; nz++){
          if(nx*nx + ny*ny + nz*nz != length2) continue;
          for (int tz : tmp) {
            for (int sz : {1,-1}) {
              double occ = 1.0;
              if(tz == -1 and sz == 1 and nx*nx + ny*ny + nz*nz > L2Fermi_pu) occ = 0.0;
              if(tz == -1 and sz ==-1 and nx*nx + ny*ny + nz*nz > L2Fermi_pd) occ = 0.0;
              if(tz ==  1 and sz == 1 and nx*nx + ny*ny + nz*nz > L2Fermi_nu) occ = 0.0;
              if(tz ==  1 and sz ==-1 and nx*nx + ny*ny + nz*nz > L2Fermi_nd) occ = 0.0;
              AddToSPStates(nx, ny, nz, sz, occ, tz);
            }
          }
        }
      }
    }
  }

}

void ModelSpace::AddToSPStates(int nx, int ny, int nz, int sz, double occ, int tz)
{
  SPState s_p = SPState(nx, ny, nz, sz, tz, occ, SPStates.size());
  for (auto & it: SPStates){
    if(it == s_p) {
      std::cout << "Warning: in " << __LINE__ << " in " << __FILE__ << s_p << " is already in the orbit list, I won't do anything" << std::endl;
      return;
    }
  }
  SPStates.push_back(s_p);
}

void ModelSpace::PrintSPStates(std::optional<int> imin, std::optional<int> imax)
{
  int min, max;
  if(imin) { min = *imin;}
  else { min = 0;}
  if(imax) { max = *imax;}
  else { max = SPStates.size();}
  std::cout << "# Single-particle states with Nmax=" << Nmax << " truncation" << std::endl;
  if(imin != 0) std::cout << "..." << std::endl;
  for (int i = min; i < max; i++){
    std::cout << GetSPState(i) << std::endl;
  }
  if(imax != SPStates.size()) std::cout << "..." << std::endl;
}
