#include "NNPWtoGrid.hh"
#include "BasicFunctions.hh"
#include "NdSpline.hh"

NNPWtoGrid::NNPWtoGrid(double Lbox, int Nmax)
  : Lbox(Lbox), Nmax(Nmax), spin_pn_channels(SpinPNChannels(Lbox,Nmax))
{
  for(int ichbra = 0; ichbra<spin_pn_channels.GetNumberChannels(); ichbra++){
    SpinPNChannel& chbra = spin_pn_channels.GetChannel(ichbra);
    for(int ichket = ichbra; ichket<spin_pn_channels.GetNumberChannels(); ichket++){
      SpinPNChannel& chket = spin_pn_channels.GetChannel(ichket);
      if(chbra.Tz != chket.Tz) continue;
      Matrices.insert({{ichbra, ichket}, NNCartesianChannel(chbra, chket)});
    }
  }
}

std::complex<double> NNPWtoGrid::GetME(std::array<int,5> ip, std::array<int,5> iq, std::array<int,5> ir, std::array<int,5> is)
{
  double phase = 1.0;
  if(ip[4]==1 and iq[4]==-1) {
    std::swap(ip, iq);
    phase *= -1.0;
  }
  if(ir[4]==1 and is[4]==-1) {
    std::swap(ir, is);
    phase *= -1.0;
  }
  int px=ip[0], py=ip[1], pz=ip[2], sp=ip[3], tp=ip[4];
  int qx=iq[0], qy=iq[1], qz=iq[2], sq=iq[3], tq=iq[4];
  int rx=ir[0], ry=ir[1], rz=ir[2], sr=ir[3], tr=ir[4];
  int sx=is[0], sy=is[1], sz=is[2], ss=is[3], ts=is[4];
  return phase * GetME(px-qx, py-qy, pz-qz, sp, sq, (tp+tq)/2, rx-sx, ry-sy, rz-sz, sr, ss, (tr+ts)/2);
}

std::complex<double> NNPWtoGrid::GetME(int ix, int iy, int iz, int si1, int si2, int ti, int jx, int jy, int jz, int sj1, int sj2, int tj)
{
  MomentumGrids& grid = spin_pn_channels.mom;
  if( std::abs(ix) > 2*grid.Nmax or
      std::abs(iy) > 2*grid.Nmax or
      std::abs(iz) > 2*grid.Nmax or
      std::abs(jx) > 2*grid.Nmax or
      std::abs(jy) > 2*grid.Nmax or
      std::abs(jz) > 2*grid.Nmax) {
    std::cout << "Error: increase Nmax" << std::endl;
    exit(0);
  }
  int ichbra = spin_pn_channels.GetChannelIndex(si1, si2, ti);
  int ichket = spin_pn_channels.GetChannelIndex(sj1, sj2, tj);
  int ibra = grid.GetGridIndex(ix, iy, iz);
  int iket = grid.GetGridIndex(jx, jy, jz);
  if(ichbra > ichket){
    return std::conj(Matrices.at({ichket,ichbra}).matrix(iket,ibra));
  }
  else{
    return Matrices.at({ichbra,ichket}).matrix(ibra,iket);
  }
}

void NNPWtoGrid::SetNNInteraction(std::string input_nn_file)
{
  std::cout << "Computing NN matrix elements from " << input_nn_file << std::endl;
  NNPartialWave nnpw = NNPartialWave(input_nn_file);
  nnpw.FixPhase();

  Eigen::VectorXd interpolants;
  Eigen::VectorXi grid_to_interpolants;
  FindInterpolants(interpolants, grid_to_interpolants, nnpw);

  NNPartialWave nnpw_new = NNPartialWave();
  nnpw_new.NChan = nnpw.NChan;
  nnpw_new.Jmax = nnpw.Jmax;
  nnpw_new.Weighted = nnpw.Weighted;
  nnpw_new.JLLSTz = nnpw.JLLSTz;
  nnpw_new.ChannelIndex = nnpw.ChannelIndex;
  nnpw_new.Mesh = interpolants;
  nnpw_new.NMesh = interpolants.size();
  nnpw_new.Weight.resize(interpolants.size());
  nnpw_new.Weight.setConstant(0.0);
  for(int idx=0; idx<nnpw_new.GetNumberChannels(); idx++){
    nnpw_new.Vmat.push_back(Eigen::MatrixXd(nnpw_new.NMesh, nnpw_new.NMesh));
  }

  Interpolate(nnpw_new, nnpw);

  store_spin_cg arr_spin_cg = store_spin_cg();
  store_lsj_cg arr_lsj_cg = store_lsj_cg(nnpw_new.Jmax);
  store_Ylm arr_Ylm = store_Ylm(spin_pn_channels.mom, nnpw_new.Jmax+1);
  ToGridBasis(nnpw_new, grid_to_interpolants, arr_spin_cg, arr_lsj_cg, arr_Ylm);
}

void NNPWtoGrid::ToGridBasis(NNPartialWave& nnpw, Eigen::VectorXi& grid_to_interpolants, store_spin_cg& spin_cg, store_lsj_cg& lsj_cg, store_Ylm& Ylm)
{
  MomentumGrids& grid = spin_pn_channels.mom;
  //double fact = std::pow(2.0 * Constants::PI / (Lbox * Constants::HBARC), 3);
  double fact = std::pow(2.0 * Constants::PI / Lbox, 3);

  std::vector<std::array<int,2>> looplist;
  for(auto& it : Matrices){
    looplist.push_back(it.first);
  }

#pragma omp parallel for schedule(dynamic)
  for(int ch_index=0; ch_index<looplist.size(); ch_index++){
    int ichbra = looplist[ch_index][0];
    int ichket = looplist[ch_index][1];
    NNCartesianChannel& ChMat = Matrices.at({ichbra,ichket});

    SpinPNChannel& chbra = spin_pn_channels.channels[ichbra];
    SpinPNChannel& chket = spin_pn_channels.channels[ichket];
    int ms1 = (chbra.sz1 + chbra.sz2)/2;
    int ms2 = (chket.sz1 + chket.sz2)/2;

    for(int ibra=0; ibra<grid.GetNumberGrids(); ibra++){
      int ip_bra = grid_to_interpolants(ibra);
      for(int iket=0; iket<grid.GetNumberGrids(); iket++){
        int ip_ket = grid_to_interpolants(iket);

        std::complex<double> res = {0.0, 0.0};
        for(std::array<int,5> arr : nnpw.JLLSTz){
          int J = arr[0];
          int L = arr[1];
          int Lp= arr[2];
          int S = arr[3];
          int Tz= arr[4];
          int idx = nnpw.GetChannelIndex(J, L, Lp, S, Tz);
          if(S < std::abs(ms1) or S < std::abs(ms2)) continue;
          if(Tz != chbra.Tz) continue;
          if(Tz != chket.Tz) continue;

          std::complex<double> orbital = {0.0, 0.0};
          for(int M=-J; M<=J; M++){
            int ml1 = M-ms1;
            int ml2 = M-ms2;
            if(L < std::abs(ml1) or Lp < std::abs(ml2)) continue;

            //orbital += BasicFunctions::CG(2*L, 2*ml1, 2*S, 2*ms1, 2*J, 2*M) * BasicFunctions::CG(2*Lp, 2*ml2, 2*S, 2*ms2, 2*J, 2*M)
            //  * BasicFunctions::Ylm(L, ml1, theta1, phi1) * std::conj(BasicFunctions::Ylm(Lp, ml2, theta2, phi2));
            orbital += lsj_cg.at(L, ml1, S, ms1, J, M) * lsj_cg.at(Lp, ml2, S, ms2, J, M)
              * Ylm.at(ibra, L, ml1) * std::conj(Ylm.at(iket, Lp, ml2));

          }
          //double spin = BasicFunctions::CG(1, chbra.sz1, 1, chbra.sz2, 2*S, 2*ms1) * BasicFunctions::CG(1, chket.sz1, 1, chket.sz2, 2*S, 2*ms2);
          double spin = spin_cg.at(chbra.sz1, chbra.sz2, S, ms1) * spin_cg.at(chket.sz1, chket.sz2, S, ms2);
          res += nnpw.Vmat[idx](ip_bra, ip_ket) * spin * orbital * (1.0 + std::abs(chbra.Tz) * std::pow(-1.0, L + S));
        }
        ChMat.matrix(ibra, iket) = res * fact;
      }
    }
  }
}

void NNPWtoGrid::Interpolate(NNPartialWave& nnpw_new, NNPartialWave& nnpw_old)
{
  int k = 4; // spline interpolation order
  int N_old = nnpw_old.NMesh;
  int N_new = nnpw_new.NMesh;
  Eigen::VectorXi ks(2), ns_old(2), ns_new(2);
  Eigen::VectorXd xs_old(2*N_old), xs_new(2*N_new);
  ks << k, k;
  ns_old << N_old, N_old;
  ns_new << N_new, N_new;
  xs_old << nnpw_old.Mesh, nnpw_old.Mesh;
  xs_new << nnpw_new.Mesh, nnpw_new.Mesh;

  for(auto& it : nnpw_new.ChannelIndex){
    int J = it.first[0];
    int L = it.first[1];
    int Lp= it.first[2];
    int S = it.first[3];
    int Tz= it.first[4];
    auto& mat_new = nnpw_new.Vmat[it.second];
    auto& mat_old = nnpw_old.Vmat[nnpw_old.ChannelIndex.at({J,L,Lp,S,Tz})];

    Eigen::Map<Eigen::VectorXd> v_original(mat_old.data(), N_old*N_old);
    NdSpline sp = NdSpline(ks, ns_old, xs_old, v_original);
    sp.Interpolate(ns_new, xs_new);
    Eigen::MatrixXd mat_interp = Eigen::Map<const Eigen::MatrixXd>(sp.fs.data(), N_new, N_new);
    mat_new = mat_interp;
  }
}

void NNPWtoGrid::FindInterpolants(Eigen::VectorXd& interpolants, Eigen::VectorXi& grid_to_interpolants, NNPartialWave& nnpw)
{
  MomentumGrids& grid = spin_pn_channels.mom;
  Eigen::VectorXd p_tmp(grid.GetNumberGrids());
  for(int idx=0; idx<grid.GetNumberGrids(); idx++){
    p_tmp(idx) = grid.grids[idx].p;
  }

  grid_to_interpolants.resize(grid.GetNumberGrids());
  int counter = 0;
  for(int idx=0; idx<grid.GetNumberGrids(); idx++){
    double pp = p_tmp.minCoeff();
    if(pp == 1.e100) continue;
    for(int idx_tmp=0; idx_tmp<grid.GetNumberGrids(); idx_tmp++){
      if(p_tmp(idx_tmp) == 1.e100) continue;
      if(pp == p_tmp(idx_tmp)){
        grid_to_interpolants(idx_tmp) = counter;
        p_tmp(idx_tmp) = 1.e100;
      }
    }
    counter += 1;
  }
  interpolants.resize(counter);
  for(int idx=0; idx<counter; idx++){
    for(int idx_tmp=0; idx_tmp<grid.GetNumberGrids(); idx_tmp++){
      if(idx == grid_to_interpolants(idx_tmp)){
        interpolants(idx) = grid.grids[idx_tmp].p;
      }
    }
  }

}

NNPWtoGrid::MomentumGrids::MomentumGrids(double Lbox, int Nmax)
  : Lbox(Lbox), Nmax(Nmax)
{
  int nx = 4*Nmax+1;
  int ny = nx;
  int nz = nx;
  for(int ix = -nx/2; ix<=nx/2; ix++){
    abscissa[ix] = Constants::PI * (double)ix / Lbox;
  }
  int idx = 0;
  for(int ix = -nx/2; ix<=nx/2; ix++){
    for(int iy = -ny/2; iy<=ny/2; iy++){
      for(int iz = -nz/2; iz<=nz/2; iz++){
        double px = abscissa[ix];
        double py = abscissa[iy];
        double pz = abscissa[iz];
        double p = std::sqrt(px*px + py*py + pz*pz);
        double theta, phi;
        if(ix==0 and iy==0 and iz==0){
          theta = 0.0;
        }
        else{
          theta = std::acos(pz / p);
        }
        if(ix==0 and iy==0){
          phi = 0.0;
        }
        else{
          phi = sign(py) * std::acos(px / std::sqrt(px*px + py*py));
        }
        grids.emplace_back(ix, iy, iz, px, py, pz, p, theta, phi);
        grid_to_idx[{ix,iy,iz}] = idx;
        idx += 1;
      }
    }
  }
}

NNPWtoGrid::SpinPNChannels::SpinPNChannels(double Lbox, int Nmax)
  : mom(MomentumGrids(Lbox, Nmax))
{
  int idx = 0;
  for(int sz1 : {-1, 1}){
    for(int sz2 : {-1, 1}){
      for(int Tz : {-1,0,1}){
        channels.emplace_back(sz1, sz2, Tz, mom);
        spin_pn_idx[{sz1, sz2, Tz}] = idx;
        idx += 1;
      }
    }
  }
}

NNPWtoGrid::NNCartesianChannel::NNCartesianChannel(SpinPNChannel& chbra, SpinPNChannel& chket)
  : channel_bra(&chbra), channel_ket(&chket), is_zero(true)
{
  matrix.resize(channel_bra->mom->GetNumberGrids(), channel_ket->mom->GetNumberGrids());
  matrix.setConstant({0.0,0.0});
}

NNPWtoGrid::store_spin_cg::store_spin_cg()
  : size_m1(2), size_m2(2), size_M(3), size_S(2)
{
  total_size = size_m1 * size_m2 * size_M * size_S;
  arr.resize(total_size);
  for(int m1 : {-1, 1}){
    for(int m2 : {-1, 1}){
      for(int S : {0, 1}){
        for(int M : {-1, 0, 1}){
          arr[cache(m1,m2,S,M)] = BasicFunctions::CG(1, m1, 1, m2, 2*S, 2*M);
        }
      }
    }
  }
}

int NNPWtoGrid::store_spin_cg::cache(int m1, int m2, int S, int Ms)
{
  int idx_m1 = (m1==-1) ? 0 : 1;
  int idx_m2 = (m2==-1) ? 0 : 1;
  int idx_S = S;
  int idx_M = Ms + 1;
  return ((idx_m1 * size_m2 + idx_m2) * size_S + idx_S) * size_M + idx_M;
}

NNPWtoGrid::store_lsj_cg::store_lsj_cg(int Jmax)
  : Jmax(Jmax), size_l(Jmax+2), size_ml(2*(Jmax+1)+1), size_s(2), size_ms(3), size_j(Jmax+1), size_m(2*Jmax+1)
{
  total_size = size_l * size_ml * size_s * size_ms * size_j * size_m;
  arr.resize(total_size);
  for(int l = 0; l<=Jmax+1; l++){
    for(int ml = -(Jmax+1); ml <= Jmax+1; ml++){
      for(int s : {0, 1}){
        for(int ms : {-1, 0, 1}){
          for(int j = 0; j<=Jmax; j++){
            for(int m=-Jmax; m<=Jmax; m++){
              arr[cache(l,ml,s,ms,j,m)] = BasicFunctions::CG(2*l, 2*ml, 2*s, 2*ms, 2*j, 2*m);
            }
          }
        }
      }
    }
  }
}

int NNPWtoGrid::store_lsj_cg::cache(int l, int ml, int s, int ms, int j, int m)
{
  int idx_l = l;
  int idx_ml = ml + Jmax+1;
  int idx_s = s;
  int idx_ms = ms + 1;
  int idx_j = j;
  int idx_m = m + Jmax;
  return ((((idx_l * size_ml + idx_ml) * size_s + idx_s) * size_ms + idx_ms) * size_j + idx_j) * size_m + idx_m;
}

NNPWtoGrid::store_Ylm::store_Ylm(MomentumGrids& g, int Lmax)
  : Lmax(Lmax), size_idx(g.GetNumberGrids()), size_l(Lmax+1), size_m(2*Lmax+1)
{
  total_size = size_idx * size_l * size_m;
  arr.resize(total_size);
  for(int idx = 0; idx<g.GetNumberGrids(); idx++){
    double theta = g.grids[idx].theta;
    double phi = g.grids[idx].phi;
    for(int l = 0; l <= Lmax; l++){
      for(int m = -Lmax; m<= Lmax; m++){
        if(std::abs(m) > l) {
          arr[cache(idx,l,m)] = {0.0, 0.0};
          continue;
        }
        arr[cache(idx,l,m)] = BasicFunctions::Ylm(l, m, theta, phi);
      }
    }
  }
}

int NNPWtoGrid::store_Ylm::cache(int idx, int l, int m)
{
  int idx_idx = idx;
  int idx_l = l;
  int idx_m = m + Lmax;
  return (idx_idx * size_l + idx_l) * size_m + idx_m;
}

