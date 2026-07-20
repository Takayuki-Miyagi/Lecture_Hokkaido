#include "Constants.hh"
#include "TwoBodyOperator.hh"
#include "ChEFT.hh"

TwoBodyOperator::~TwoBodyOperator()
{}

TwoBodyOperator::TwoBodyOperator()
{}

TwoBodyOperator& TwoBodyOperator::operator*=(const double rhs)
{
  for ( auto& itmat : Matrices ) {
    itmat.second *= rhs;
  }
  XValid = false;
  return *this;
}

TwoBodyOperator& TwoBodyOperator::operator+=(const TwoBodyOperator& rhs)
{
  for ( auto& itmat : rhs.Matrices ) {
    int ch_bra = itmat.first[0];
    int ch_ket = itmat.first[1];
    Matrices.at({ch_bra,ch_ket}) += itmat.second;
  }
  XValid = false;
  return *this;
}

TwoBodyOperator& TwoBodyOperator::operator-=(const TwoBodyOperator& rhs)
{
  for ( auto& itmat : rhs.Matrices ) {
    int ch_bra = itmat.first[0];
    int ch_ket = itmat.first[1];
    Matrices.at({ch_bra,ch_ket}) -= itmat.second;
  }
  XValid = false;
  return *this;
}

TwoBodyOperator::TwoBodyOperator(ModelSpace& modelspace, int rankTz, int Qx, int Qy, int Qz):
  modelspace(&modelspace), rankTz(rankTz), Qx(Qx), Qy(Qy), Qz(Qz)
{
  for(auto & it_bra: modelspace.TwoBody.Channels){
    for(auto & it_ket: modelspace.TwoBody.Channels){
      if(it_bra.ChannelIndex > it_ket.ChannelIndex) continue;
      if(it_bra.Nx - it_ket.Nx != Qx) continue;
      if(it_bra.Ny - it_ket.Ny != Qy) continue;
      if(it_bra.Nz - it_ket.Nz != Qz) continue;
      if(it_bra.Tz - it_ket.Tz != rankTz) continue;
      Matrices[{it_bra.ChannelIndex, it_ket.ChannelIndex}].resize(it_bra.GetNumberStates(), it_ket.GetNumberStates());
      Matrices.at({it_bra.ChannelIndex, it_ket.ChannelIndex}).setConstant({0.0,0.0});
    }
  }
}

void TwoBodyOperator::SetME(int ip, int iq, int ir, int is, std::complex<double> me)
{
  if(std::abs(me) < 1.e-16) return;
  if(ip==iq or ir==is) return;
  SPState& p = modelspace->GetSPState(ip);
  SPState& q = modelspace->GetSPState(iq);
  SPState& r = modelspace->GetSPState(ir);
  SPState& s = modelspace->GetSPState(is);
  if((p.nx + q.nx - r.nx - s.nx) - Qx != 0) return;
  if((p.ny + q.ny - r.ny - s.ny) - Qy != 0) return;
  if((p.nz + q.nz - r.nz - s.nz) - Qz != 0) return;
  if((p.tz + q.tz - r.tz - s.tz)/2 - rankTz != 0) return;
  int ichbra = modelspace->TwoBody.GetChannelIndex((p.tz+q.tz)/2, p.nx+q.nx, p.ny+q.ny, p.nz+q.nz);
  int ichket = modelspace->TwoBody.GetChannelIndex((r.tz+s.tz)/2, r.nx+s.nx, r.ny+s.ny, r.nz+s.nz);
  double phase = 1.0;
  if(ichbra > ichket){
    std::swap(ichbra, ichket);
    std::swap(ip,ir);
    std::swap(iq,is);
    if(AntiSym) phase *= -1.0;
    me = std::conj(me);
  }
  auto& Matrix = Matrices.at({ichbra,ichket});
  auto& tbcbra = modelspace->TwoBody.GetChannel(ichbra);
  auto& tbcket = modelspace->TwoBody.GetChannel(ichket);
  int idxbra = tbcbra.GetIndex(ip,iq);
  int idxket = tbcket.GetIndex(ir,is);
  int phbra = tbcbra.GetPhase(ip,iq);
  int phket = tbcket.GetPhase(ir,is);
  if(std::abs(Matrix(idxbra,idxket)) > 1.e-16){
    std::cout << "Warning in SetME; a nonzero me existss. Please make sure if it is ok: " << std::endl;
    std::cout << __LINE__ << " " << __FILE__ << std::endl;
  }
  Matrix(idxbra, idxket) = me * (double)(phbra*phket) * phase;
  if(ichbra!=ichket) return;
  if(AntiSym) phase *= -1.0;
  Matrix(idxket, idxbra) = std::conj(me) * (double)(phbra*phket) * phase;
  XValid = false;
}

std::complex<double> TwoBodyOperator::GetME(int ip, int iq, int ir, int is) const
{
  if(ip==iq or ir==is) return {0.0, 0.0};
  const SPState& p = modelspace->GetSPState(ip);
  const SPState& q = modelspace->GetSPState(iq);
  const SPState& r = modelspace->GetSPState(ir);
  const SPState& s = modelspace->GetSPState(is);
  if((p.nx + q.nx - r.nx - s.nx) - Qx != 0) return {0.0, 0.0};
  if((p.ny + q.ny - r.ny - s.ny) - Qy != 0) return {0.0, 0.0};
  if((p.nz + q.nz - r.nz - s.nz) - Qz != 0) return {0.0, 0.0};
  if((p.tz + q.tz - r.tz - s.tz)/2 - rankTz != 0) return {0.0, 0.0};
  int ichbra = modelspace->TwoBody.GetChannelIndex((p.tz+q.tz)/2, p.nx+q.nx, p.ny+q.ny, p.nz+q.nz);
  int ichket = modelspace->TwoBody.GetChannelIndex((r.tz+s.tz)/2, r.nx+s.nx, r.ny+s.ny, r.nz+s.nz);
  double phase = 1.0;
  if(ichbra > ichket){
    std::swap(ichbra, ichket);
    std::swap(ip,ir);
    std::swap(iq,is);
    if(AntiSym) phase *= -1.0;
  }
  const auto& Matrix = Matrices.at({ichbra,ichket});

  auto& tbcbra = modelspace->TwoBody.GetChannel(ichbra);
  auto& tbcket = modelspace->TwoBody.GetChannel(ichket);
  int idxbra = tbcbra.GetIndex(ip,iq);
  int idxket = tbcket.GetIndex(ir,is);
  int phbra = tbcbra.GetPhase(ip,iq);
  int phket = tbcket.GetPhase(ir,is);
  return Matrix(idxbra, idxket) * (double)(phbra*phket) * phase;
}

void TwoBodyOperator::SetNNInteraction(int ch_order, std::map<std::string,double> LECs, double RegLambda, int reg_power, double SFRLambda, bool chiral_delta)
{
  double start_time = omp_get_wtime();
  //ChEFT::ChiralInteraction nnint = ChEFT::ChiralInteraction(*this->modelspace);
  ChEFT::ChiralInteraction nnint = ChEFT::ChiralInteraction(*modelspace);
  nnint.SetChiralOrder(ch_order);
  nnint.SetLECs(LECs);
  nnint.SetLambdaCut(RegLambda);
  nnint.SetLambdaSFR(SFRLambda);
  nnint.SetRegPower(reg_power);
  nnint.SetDelta(chiral_delta);

#pragma omp parallel for schedule(dynamic)
  for(int ich=0; ich < modelspace->TwoBody.GetNumberChannels(); ich++){
    TwoBodyChannel& tbc = modelspace->TwoBody.GetChannel(ich);
    for(int idx_bra=0; idx_bra < tbc.GetNumberStates(); idx_bra++){
      int ip = tbc.GetSPStateIndex1(idx_bra);
      int iq = tbc.GetSPStateIndex2(idx_bra);
      for(int idx_ket=idx_bra; idx_ket < tbc.GetNumberStates(); idx_ket++){
        int ir = tbc.GetSPStateIndex1(idx_ket);
        int is = tbc.GetSPStateIndex2(idx_ket);
        std::complex<double> me = nnint.CalcNNAsymME(ip, iq, ir, is);
        SetME(ip, iq, ir, is, me);
      }
    }
  }
  Profiler::timer[__func__] += omp_get_wtime() - start_time;
}

void TwoBodyOperator::SetNNInteraction(std::string input_nn_file)
{
  double start_time = omp_get_wtime();

  NNPWtoGrid nng = NNPWtoGrid(modelspace->Lbox, modelspace->Nmax);
  nng.SetNNInteraction(input_nn_file);

#pragma omp parallel for schedule(dynamic)
  for(int ich=0; ich < modelspace->TwoBody.GetNumberChannels(); ich++){
    TwoBodyChannel& tbc = modelspace->TwoBody.GetChannel(ich);
    for(int idx_bra=0; idx_bra < tbc.GetNumberStates(); idx_bra++){
      int ip = tbc.GetSPStateIndex1(idx_bra);
      int iq = tbc.GetSPStateIndex2(idx_bra);
      SPState& p = modelspace->GetSPState(ip);
      SPState& q = modelspace->GetSPState(iq);
      std::array<int,5> pp = {p.nx, p.ny, p.nz, p.sz, p.tz};
      std::array<int,5> qq = {q.nx, q.ny, q.nz, q.sz, q.tz};
      for(int idx_ket=idx_bra; idx_ket < tbc.GetNumberStates(); idx_ket++){
        int ir = tbc.GetSPStateIndex1(idx_ket);
        int is = tbc.GetSPStateIndex2(idx_ket);
        SPState& r = modelspace->GetSPState(ir);
        SPState& s = modelspace->GetSPState(is);
        std::array<int,5> rr = {r.nx, r.ny, r.nz, r.sz, r.tz};
        std::array<int,5> ss = {s.nx, s.ny, s.nz, s.sz, s.tz};

        std::complex<double> me = nng.GetME(pp, qq, rr, ss);
        SetME(ip, iq, ir, is, me);
      }
    }
  }
  Profiler::timer[__func__] += omp_get_wtime() - start_time;
}

void TwoBodyOperator::SetNO2B3NInteraction(int ch_order, std::map<std::string,double> LECs, double RegLambda, int reg_power, double SFRLambda, bool chiral_delta)
{
  double start_time = omp_get_wtime();
  ChEFT::ChiralInteraction nnnint = ChEFT::ChiralInteraction(*this->modelspace);
  nnnint.SetChiralOrder(ch_order);
  nnnint.SetLECs(LECs);
  nnnint.SetLambdaCut(RegLambda);
  nnnint.SetLambdaSFR(SFRLambda);
  nnnint.SetRegPower(reg_power);
  nnnint.SetDelta(chiral_delta);

#pragma omp parallel for schedule(dynamic)
  for(int ich=0; ich < modelspace->TwoBody.GetNumberChannels(); ich++){
    TwoBodyChannel& tbc = modelspace->TwoBody.GetChannel(ich);
    for(int idx_bra=0; idx_bra < tbc.GetNumberStates(); idx_bra++){
      int ip = tbc.GetSPStateIndex1(idx_bra);
      int iq = tbc.GetSPStateIndex2(idx_bra);
      for(int idx_ket=idx_bra; idx_ket < tbc.GetNumberStates(); idx_ket++){
        int ir = tbc.GetSPStateIndex1(idx_ket);
        int is = tbc.GetSPStateIndex2(idx_ket);
        std::complex<double> me = {0.0, 0.0};
        for(int it : modelspace->hole_states){
          me += nnnint.Calc3NAsymME(ip, iq, it, ir, is, it);
        }
        SetME(ip, iq, ir, is, me);
      }
    }
  }
  Profiler::timer[__func__] += omp_get_wtime() - start_time;
}

void TwoBodyOperator::PrintOperator(std::string by) const
{
  if(by=="matrix") {
    for ( auto& itmat : Matrices )
    {
      std::cout << itmat.first[0] << " " << itmat.first[1] << std::endl;
      std::cout << itmat.second << std::endl;
    }
  }
  else if(by=="element") {
    for ( auto& itmat : Matrices )
    {
      TwoBodyChannel& tbcbra = modelspace->TwoBody.GetChannel(itmat.first[0]);
      TwoBodyChannel& tbcket = modelspace->TwoBody.GetChannel(itmat.first[1]);
      for(int idx_bra=0; idx_bra<tbcbra.GetNumberStates(); idx_bra++){
        int ip = tbcbra.GetSPStateIndex1(idx_bra);
        int iq = tbcbra.GetSPStateIndex2(idx_bra);
        SPState& p = modelspace->GetSPState(ip);
        SPState& q = modelspace->GetSPState(iq);
        for(int idx_ket=0; idx_ket<tbcket.GetNumberStates(); idx_ket++){
          int ir = tbcket.GetSPStateIndex1(idx_ket);
          int is = tbcket.GetSPStateIndex2(idx_ket);
          SPState& r = modelspace->GetSPState(ir);
          SPState& s = modelspace->GetSPState(is);
          std::complex<double> me = itmat.second(idx_bra,idx_ket);
          std::cout
            << "(" << std::setw(3) << tbcbra.Tz << std::setw(3) << tbcbra.Nx << std::setw(3) << tbcbra.Ny << std::setw(3) << tbcbra.Nz << ") "
            << "(" << std::setw(3) << tbcbra.Tz << std::setw(3) << tbcbra.Nx << std::setw(3) << tbcbra.Ny << std::setw(3) << tbcbra.Nz << ") "
            << "(" << std::setw(3) << p.nx << std::setw(3) << p.ny << std::setw(3) << p.nz << std::setw(3) << p.sz << std::setw(3) << p.tz << ") "
            << "(" << std::setw(3) << q.nx << std::setw(3) << q.ny << std::setw(3) << q.nz << std::setw(3) << q.sz << std::setw(3) << q.tz << ") "
            << "(" << std::setw(3) << r.nx << std::setw(3) << r.ny << std::setw(3) << r.nz << std::setw(3) << r.sz << std::setw(3) << r.tz << ") "
            << "(" << std::setw(3) << s.nx << std::setw(3) << s.ny << std::setw(3) << s.nz << std::setw(3) << s.sz << std::setw(3) << s.tz << ") "
            //<< "(" << std::setw(3) << p.nx-q.nx << std::setw(3) << p.ny-q.ny << std::setw(3) << p.nz-q.nz << std::setw(3) << p.sz+q.sz << std::setw(3) << p.tz+q.tz << ") "
            //<< "(" << std::setw(3) << r.nx-s.nx << std::setw(3) << r.ny-s.ny << std::setw(3) << r.nz-s.nz << std::setw(3) << r.sz+s.sz << std::setw(3) << r.tz+s.tz << ") "
            << std::setw(16) << std::setprecision(8) << std::fixed << std::real(me)
            << std::setw(16) << std::setprecision(8) << std::fixed << std::imag(me) << std::endl;
        }
      }
    }
  }
}

double TwoBodyOperator::Norm() const
{
  double res = 0.0;
  for(auto& itmat : Matrices){
    res += std::real((itmat.second * itmat.second.adjoint()).trace());
  }
  return res;
}

void TwoBodyOperator::AllocateXOperator() const
{
  double start_time = omp_get_wtime();

  if (XAllocated) {
    Profiler::timer[__func__] += omp_get_wtime() - start_time;
    return;
  }

  for (auto & it_bra: modelspace->TwoBody.XChannels) {
    for (auto & it_ket: modelspace->TwoBody.XChannels) {
      if (it_bra.nx - it_ket.nx != Qx) continue;
      if (it_bra.ny - it_ket.ny != Qy) continue;
      if (it_bra.nz - it_ket.nz != Qz) continue;
      if (it_bra.Tz - it_ket.Tz != rankTz) continue;
      XMatrices[{it_bra.ChannelIndex, it_ket.ChannelIndex}].resize(it_bra.GetNumberStates(), it_ket.GetNumberStates());
      XMatrices.at({it_bra.ChannelIndex, it_ket.ChannelIndex}).setConstant({0.0, 0.0});
    }
  }

  XAllocated = true;
  XValid     = false;
  Profiler::timer[__func__] += omp_get_wtime() - start_time;
}

// <pq|O|rs> --> <p\bar{s}|O|r\bar{q}>
void TwoBodyOperator::SetXOperator() const
{
  double start_time = omp_get_wtime();
  std::vector<std::array<int,2>> looplist;
  for(auto& it : XMatrices){
    looplist.push_back(it.first);
  }

#pragma omp parallel for schedule(dynamic)
  for(int chindices=0; chindices<looplist.size(); chindices++){
    int ichbra = looplist[chindices][0];
    int ichket = looplist[chindices][1];
    XTwoBodyChannel & xtbcbra = modelspace->TwoBody.GetXChannel(ichbra);
    XTwoBodyChannel & xtbcket = modelspace->TwoBody.GetXChannel(ichket);
    auto& mat = XMatrices.at({ichbra,ichket});
    for(int idx_bra=0; idx_bra < xtbcbra.GetNumberStates(); idx_bra++){
      int ip = xtbcbra.GetSPStateIndex1(idx_bra);
      int iq = xtbcbra.GetSPStateIndex2(idx_bra);

      int idx_ket_min = 0;
      if(ichbra==ichket) idx_ket_min = idx_bra;
      for(int idx_ket=idx_ket_min; idx_ket < xtbcket.GetNumberStates(); idx_ket++){
        int ir = xtbcket.GetSPStateIndex1(idx_ket);
        int is = xtbcket.GetSPStateIndex2(idx_ket);
        mat(idx_bra, idx_ket) = GetME(ip, is, ir, iq);
        if(ichbra!=ichket) continue;
        if(AntiSym) {
          mat(idx_ket, idx_bra) = std::conj(mat(idx_bra, idx_ket)) * -1.0;;
        }
        else {
          mat(idx_ket, idx_bra) = std::conj(mat(idx_bra, idx_ket));
        }
      }
    }
  }
  Profiler::timer[__func__] += omp_get_wtime() - start_time;
  XValid = true;
}

// <p\bar{s}|O|r\bar{q}> --> <pq|O|rs>
void TwoBodyOperator::UpdateFromXOperator()
{
  double start_time = omp_get_wtime();

  std::vector<std::array<int,2>> looplist;
  for(auto& it : Matrices){
    looplist.push_back(it.first);
  }
#pragma omp parallel for schedule(dynamic)
  for(int chindices=0; chindices<looplist.size(); chindices++){
    int ichbra = looplist[chindices][0];
    int ichket = looplist[chindices][1];
    TwoBodyChannel & tbcbra = modelspace->TwoBody.GetChannel(ichbra);
    TwoBodyChannel & tbcket = modelspace->TwoBody.GetChannel(ichket);

    auto& mat = Matrices.at({ichbra,ichket});
    for(int idx_bra=0; idx_bra < tbcbra.GetNumberStates(); idx_bra++){
      int ip = tbcbra.GetSPStateIndex1(idx_bra);
      int iq = tbcbra.GetSPStateIndex2(idx_bra);

      int idx_ket_min = 0;
      if(ichbra==ichket) idx_ket_min = idx_bra;
      for(int idx_ket=idx_ket_min; idx_ket < tbcket.GetNumberStates(); idx_ket++){
        int ir = tbcket.GetSPStateIndex1(idx_ket);
        int is = tbcket.GetSPStateIndex2(idx_ket);
        std::complex<double> me = GetXME(ip, is, ir, iq) - GetXME(iq, is, ir, ip) - GetXME(ip, ir, is, iq) + GetXME(iq, ir, is, ip); // Antisymmtrization
        me *= 0.25;
        mat(idx_bra, idx_ket) = me;

        if(ichbra!=ichket) continue;
        if(AntiSym) {
          mat(idx_ket, idx_bra) = std::conj(mat(idx_bra, idx_ket)) * -1.0;;
        }
        else {
          mat(idx_ket, idx_bra) = std::conj(mat(idx_bra, idx_ket));
        }

      }
    }
  }
  Profiler::timer[__func__] += omp_get_wtime() - start_time;
}

std::complex<double> TwoBodyOperator::GetXME(int ip, int iq, int ir, int is) const
{
  SPState& p = modelspace->GetSPState(ip);
  SPState& q = modelspace->GetSPState(iq);
  SPState& r = modelspace->GetSPState(ir);
  SPState& s = modelspace->GetSPState(is);
  if((p.tz - q.tz - r.tz + s.tz)/2 - rankTz != 0) return {0.0, 0.0};
  if((p.nx - q.nx - r.nx + s.nx) - Qx != 0) return {0.0, 0.0};
  if((p.ny - q.ny - r.ny + s.ny) - Qy != 0) return {0.0, 0.0};
  if((p.nz - q.nz - r.nz + s.nz) - Qz != 0) return {0.0, 0.0};
  int ichbra = modelspace->TwoBody.GetXChannelIndex((p.tz-q.tz)/2, p.nx-q.nx, p.ny-q.ny, p.nz-q.nz);
  int ichket = modelspace->TwoBody.GetXChannelIndex((r.tz-s.tz)/2, r.nx-s.nx, r.ny-s.ny, r.nz-s.nz);
  const auto& Matrix = XMatrices.at({ichbra,ichket});
  auto& tbcbra = modelspace->TwoBody.GetXChannel(ichbra);
  auto& tbcket = modelspace->TwoBody.GetXChannel(ichket);
  int idxbra = tbcbra.GetIndex(ip,iq);
  int idxket = tbcket.GetIndex(ir,is);
  return Matrix(idxbra, idxket);
}

void TwoBodyOperator::Erase()
{
  for(auto & it_bra: modelspace->TwoBody.Channels){
    for(auto & it_ket: modelspace->TwoBody.Channels){
      if(it_bra.ChannelIndex > it_ket.ChannelIndex) continue;
      if(it_bra.Nx - it_ket.Nx != Qx) continue;
      if(it_bra.Ny - it_ket.Ny != Qy) continue;
      if(it_bra.Nz - it_ket.Nz != Qz) continue;
      if(it_bra.Tz - it_ket.Tz != rankTz) continue;
      Matrices[{it_bra.ChannelIndex, it_ket.ChannelIndex}].resize(it_bra.GetNumberStates(), it_ket.GetNumberStates());
      Matrices.at({it_bra.ChannelIndex, it_ket.ChannelIndex}).setConstant({0.0,0.0});
    }
  }
}

void TwoBodyOperator::Clear()
{
  Matrices.clear();
}

double TwoBodyOperator::Memory() const
{
  double memory = 0.0;
  for ( auto& itmat : Matrices )
  {
    auto& mat = itmat.second;
    memory += sizeof(std::complex<double>) * (double)mat.cols() * (double)mat.rows();
  }
  return memory; // in unit of of byte
}

double TwoBodyOperator::XMemory() const
{
  double memory = 0.0;
  for ( auto& itmat : XMatrices )
  {
    auto& mat = itmat.second;
    memory += sizeof(std::complex<double>) * (double)mat.cols() * (double)mat.rows();
  }
  return memory; // in unit of of byte
}

void TwoBodyOperator::EnsureXOperator() const
{
  if (XValid) return;

  if (!XAllocated) {
    AllocateXOperator();
    XAllocated = true;
  }

  SetXOperator(); // <pq|O|rs> -> <p\bar{s}|O|r\bar{q}>
  XValid = true;
}

void TwoBodyOperator::InvalidateXOperator()
{
  XValid = false;
}

void TwoBodyOperator::EraseXOperator() const
{
  XMatrices.clear();
  XAllocated = false;
  XValid     = false;
}
