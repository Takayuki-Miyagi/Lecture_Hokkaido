#include "Operator.hh"

Operator::~Operator()
{
  Profiler::counter["N_Operators"]--;
}

Operator::Operator(const Operator& op):
  modelspace(op.modelspace),
  rankTz(op.rankTz), Qx(op.Qx), Qy(op.Qy), Qz(op.Qz),
  ZeroBody(op.ZeroBody),
  OneBody(op.OneBody), TwoBody(op.TwoBody), normal_ordered(op.normal_ordered)
{
  Profiler::counter["N_Operators"]++;
}

Operator::Operator(Operator&& op):
  modelspace(op.modelspace),
  rankTz(op.rankTz), Qx(op.Qx), Qy(op.Qy), Qz(op.Qz),
  ZeroBody(op.ZeroBody),
  OneBody(op.OneBody), TwoBody(op.TwoBody), normal_ordered(op.normal_ordered)
{
  Profiler::counter["N_Operators"]++;
}

Operator &Operator::operator=(const Operator &op) = default;
Operator &Operator::operator=(Operator &&op) = default;

Operator& Operator::operator*=(const double rhs)
{
  ZeroBody *= rhs;
  OneBody *= rhs;
  TwoBody *= rhs;
  return *this;
}

Operator Operator::operator*(const double rhs) const
{
  Operator opout = Operator(*this);
  opout *= rhs;
  return opout;
}

Operator operator*(const double lhs, const Operator& rhs)
{
   return rhs * lhs;
}
Operator operator*(const double lhs, const Operator&& rhs)
{
   return rhs * lhs;
}

Operator& Operator::operator/=(const double rhs)
{
  return *this *=(1.0/rhs);
}

Operator Operator::operator/(const double rhs) const
{
  Operator opout = Operator(*this);
  opout *= (1.0/rhs);
  return opout;
}

Operator& Operator::operator+=(const Operator& rhs)
{
  ZeroBody += rhs.ZeroBody;
  OneBody  += rhs.OneBody;
  TwoBody  += rhs.TwoBody;
  return *this;
}

Operator Operator::operator+(const Operator& rhs) const
{
  return ( Operator(*this) += rhs );
}

Operator& Operator::operator+=(const double& rhs)
{
  ZeroBody += rhs;
  return *this;
}

Operator Operator::operator+(const double& rhs) const
{
  return ( Operator(*this) += rhs );
}

Operator& Operator::operator-=(const Operator& rhs)
{
  ZeroBody -= rhs.ZeroBody;
  OneBody -= rhs.OneBody;
  TwoBody -= rhs.TwoBody;
  return *this;
}

Operator Operator::operator-(const Operator& rhs) const
{
  return ( Operator(*this) -= rhs );
}

Operator& Operator::operator-=(const double& rhs)
{
  ZeroBody -= rhs;
  return *this;
}

Operator Operator::operator-(const double& rhs) const
{
  return ( Operator(*this) -= rhs );
}

Operator Operator::operator-() const
{
  return (*this)*-1.0;
}

Operator::Operator(ModelSpace& ms, std::optional<int> rankTz, std::optional<int> Qx, std::optional<int> Qy, std::optional<int> Qz)
  : modelspace(&ms), rankTz(*rankTz), Qx(*Qx), Qy(*Qy), Qz(*Qz)
{
  ZeroBody = 0.0;
  OneBody = OneBodyOperator(*modelspace, *rankTz, *Qx, *Qy, *Qz);
  TwoBody = TwoBodyOperator(*modelspace, *rankTz, *Qx, *Qy, *Qz);
  Profiler::counter["N_Operators"]++;
}

void Operator::Allocate(ModelSpace& ms, int rankTz, int Qx, int Qy, int Qz)
{
  modelspace = &ms;
  ZeroBody = 0.0;
  OneBody = OneBodyOperator(*modelspace, rankTz, Qx, Qy, Qz);
  TwoBody = TwoBodyOperator(*modelspace, rankTz, Qx, Qy, Qz);
}

void Operator::AllocateV3NO2B()
{
  V3NO2B = TwoBodyOperator(*modelspace, rankTz, Qx, Qy, Qz);
}

void Operator::NormalOrderingH()
{
  if(normal_ordered){
    std::cout << "Warning: operator is already normal ordered! I do nothing here." << std::endl;
    return;
  }

  double start_time = omp_get_wtime();
  ZeroBody = 0.0;
  std::complex<double> ZeroBody1 = {0.0, 0.0};
  std::complex<double> ZeroBody2 = {0.0, 0.0};
  std::complex<double> ZeroBody3 = {0.0, 0.0};
  for(int ii : modelspace->hole_states){
    SPState& i = modelspace->GetSPState(ii);
    for(int ij : modelspace->hole_states){
      SPState& j = modelspace->GetSPState(ij);
      ZeroBody2 += Get2BME(ii,ij,ii,ij) * i.occ * j.occ * 0.5;
      ZeroBody3 += V3NO2B.GetME(ii,ij,ii,ij) * i.occ * j.occ / 6.0;
    }
    ZeroBody1 += Get1BME(ii,ii) * i.occ;
  }
  std::cout << "Taking normal ordering..." << std::endl;
  std::cout << "0 body terms are 1->0: " <<
    std::fixed << std::setprecision(4) << std::setw(14) << std::real(ZeroBody1) <<
    ",  2->0: " << std::setprecision(4) << std::setw(14) << std::real(ZeroBody2) <<
    ",  3->0: " << std::setprecision(4) << std::setw(14) << std::real(ZeroBody3) << std::endl;
  ZeroBody = ZeroBody1 + ZeroBody2 + ZeroBody3;

  for(int ip : modelspace->all_states){
    for(int iq: OneBody.non_zero_loop_indices.at(ip)){
      for(int ii : modelspace->hole_states){
        SPState& i = modelspace->GetSPState(ii);
        OneBody.Matrix(ip,iq) += Get2BME(ip,ii,iq,ii) * i.occ;
        OneBody.Matrix(ip,iq) += V3NO2B.GetME(ip,ii,iq,ii) * i.occ * 0.5;
      }
    }
  }
  TwoBody += V3NO2B;
  V3NO2B.Clear();

  normal_ordered = true;
  Profiler::timer[__func__] += omp_get_wtime() - start_time;
}

void Operator::NormalOrdering()
{
  if(normal_ordered){
    std::cout << "Warning: operator is already normal ordered! I do nothing here." << std::endl;
    return;
  }

  double start_time = omp_get_wtime();
  ZeroBody = 0.0;
  std::complex<double> ZeroBody1 = {0.0, 0.0};
  std::complex<double> ZeroBody2 = {0.0, 0.0};
  for(int ii : modelspace->hole_states){
    SPState& i = modelspace->GetSPState(ii);
    for(int ij : modelspace->hole_states){
      SPState& j = modelspace->GetSPState(ij);
      ZeroBody2 += Get2BME(ii,ij,ii,ij) * i.occ * j.occ * 0.5;
    }
    ZeroBody1 += Get1BME(ii,ii) * i.occ;
  }
  std::cout << "Taking normal ordering..." << std::endl;
  std::cout << "0 body terms are 1->0: " <<
    std::fixed << std::setprecision(4) << std::setw(14) << std::real(ZeroBody1) <<
    ",  2->0: " << std::setprecision(4) << std::setw(14) << std::real(ZeroBody2) << std::endl;
  ZeroBody = ZeroBody1 + ZeroBody2;

  for(int ip : modelspace->all_states){
    for(int iq: OneBody.non_zero_loop_indices.at(ip)){
      for(int ii : modelspace->hole_states){
        SPState& i = modelspace->GetSPState(ii);
        OneBody.Matrix(ip,iq) += Get2BME(ip,ii,iq,ii) * i.occ;
      }
    }
  }
  normal_ordered = true;
  Profiler::timer[__func__] += omp_get_wtime() - start_time;
}

void Operator::SetGenerator(Operator& Hin)
{
  double start_time = omp_get_wtime();
  OneBody.Matrix.setConstant({0.0,0.0});
  TwoBody.Erase();
#pragma omp parallel for schedule(dynamic)
  for(int ich=0; ich < modelspace->TwoBody.GetNumberChannels(); ich++){
    TwoBodyChannel& tbc = modelspace->TwoBody.GetChannel(ich);
    for(int idx_bra=0; idx_bra < tbc.GetNumberStates(); idx_bra++){
      int ip = tbc.GetSPStateIndex1(idx_bra);
      int iq = tbc.GetSPStateIndex2(idx_bra);
      SPState& p = modelspace->GetSPState(ip);
      SPState& q = modelspace->GetSPState(iq);
      double occ_bra = (1.0-p.occ) * (1.0-q.occ);
      if(occ_bra < 1.e-8) continue;
      for(int idx_ket=0; idx_ket < tbc.GetNumberStates(); idx_ket++){
        int ir = tbc.GetSPStateIndex1(idx_ket);
        int is = tbc.GetSPStateIndex2(idx_ket);
        SPState& r = modelspace->GetSPState(ir);
        SPState& s = modelspace->GetSPState(is);
        double occ_ket = r.occ * s.occ;
        if(occ_ket < 1.e-8) continue;
        //std::complex<double> denom = Hin.Get1BME(ip,ip) + Hin.Get1BME(iq,iq) - Hin.Get1BME(ir,ir) - Hin.Get1BME(is,is) +
        //  Hin.Get2BME(ip,iq,ip,iq) + Hin.Get2BME(ir,is,ir,is) - Hin.Get2BME(ip,ir,ip,ir) - Hin.Get2BME(iq,is,iq,is) - Hin.Get2BME(ip,is,ip,is) - Hin.Get2BME(iq,ir,iq,ir);
        std::complex<double> denom = Hin.Get1BME(ip,ip) + Hin.Get1BME(iq,iq) - Hin.Get1BME(ir,ir) - Hin.Get1BME(is,is);
        std::complex<double> me = Hin.Get2BME(ip,iq,ir,is) / denom;
        me *= occ_bra * occ_ket;
        Set2BME(ip, iq, ir, is, me);
        //std::cout << ip << " " << iq << " " << ir << " " << is << " " << Hin.Get2BME(ip,iq,ir,is) << " " << denom << " " << Get2BME(ip,iq,ir,is) << std::endl;
      }
    }
  }
  Profiler::timer[__func__] += omp_get_wtime() - start_time;
}

double Operator::Norm() const
{
  return OneBody.Norm() + TwoBody.Norm();
}

void Operator::PrintEnergy()
{
  std::cout << "E/A (HF): " << std::fixed << std::setw(16) << std::setprecision(8) << std::real(ZeroBody) / (double)modelspace->A
    << " + " << std::scientific << std::setw(12) << std::setprecision(4) << std::imag(ZeroBody) / (double)modelspace->A << "i MeV" << std::endl;
}

void Operator::PrintSPE()
{
  std::cout << "# Single-particle energies:  " << std::endl;
  for(int L2 = 0; L2<=3*std::pow(modelspace->Nmax,2); L2++){
    for(int ip : modelspace->all_states){
      SPState& p = modelspace->GetSPState(ip);
      if(p.nx*p.nx + p.ny*p.ny + p.nz*p.nz != L2) continue;
      std::string ph = (p.occ > 1.e-8) ? "hole" : "particle";
      std::cout
        << std::setw(10) << ph << " "
        << std::setw(6) << "|k|:" << std::fixed << std::setprecision(6) << std::setw(12) <<  modelspace->GetMomentum(p).norm() << " fm-1, SPE:"
        << std::fixed << std::setprecision(8) << std::setw(16) << std::real(Get1BME(ip,ip)) << " + "
        << std::scientific << std::setprecision(4) << std::setw(12) << std::imag(Get1BME(ip,ip)) << "i MeV" << std::endl;
      break;
    }
  }
  std::cout << std::endl;
}

void Operator::PrintOperator(std::string by)
{
  std::cout << "0B: " << std::setprecision(6) << std::setw(12) << ZeroBody << std::endl;
  std::cout << "1B: " << std::endl;
  OneBody.PrintOperator();
  std::cout << "2B: " << std::endl;
  TwoBody.PrintOperator(by);

}

void Operator::Erase()
{
  ZeroBody = 0.0;
  OneBody.Erase();
  TwoBody.Erase();
}

double Operator::Memory()
{
  double memory = sizeof(ZeroBody); // zero-body part
  memory += OneBody.Memory() + TwoBody.Memory();
  return memory; // in unit of byte
}

void Operator::SetSpinOperator()
{
  OneBody.SetSpinOperator();
}
