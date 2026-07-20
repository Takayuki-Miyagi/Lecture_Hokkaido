#include "OneBodyOperator.hh"

OneBodyOperator::~OneBodyOperator()
{}

OneBodyOperator::OneBodyOperator()
{}

OneBodyOperator& OneBodyOperator::operator*=(const double rhs)
{
  Matrix *= rhs;
  return *this;
}

OneBodyOperator& OneBodyOperator::operator+=(const OneBodyOperator& rhs)
{
  Matrix += rhs.Matrix;
  return *this;
}

OneBodyOperator& OneBodyOperator::operator-=(const OneBodyOperator& rhs)
{
  Matrix -= rhs.Matrix;
  return *this;
}

OneBodyOperator::OneBodyOperator(ModelSpace& modelspace, int rankTz, int Qx, int Qy, int Qz):
  modelspace(&modelspace), rankTz(rankTz), Qx(Qx), Qy(Qy), Qz(Qz)
{
  int norb = modelspace.GetNumberSPStates();
  Matrix.resize(norb, norb);
  Matrix.setConstant({0.0,0.0});
  SetNonZeroLoopIndices();
}

void OneBodyOperator::SetKineticEnergy()
{
  for(int ip : modelspace->all_states){
    SPState& p = modelspace->GetSPState(ip);
    for(int iq : GetNonZeroLoop(ip)){
      SPState& q = modelspace->GetSPState(iq);
      if(p.sz != q.sz) continue;
      double me = modelspace->GetMomentum(p).squaredNorm() * Constants::HBARC * Constants::HBARC / (2.0 * Constants::M_NUCLEON);
      SetME(ip, iq, me);
    }
  }
}

void OneBodyOperator::SetNonZeroLoopIndices()
{
  for(int ip : modelspace->all_states){
    SPState& p = modelspace->GetSPState(ip);
    for(int iq : modelspace->all_states){
      SPState& q = modelspace->GetSPState(iq);
      if((p.tz - q.tz)/2 - rankTz != 0) continue;
      if((p.nx - q.nx) - Qx != 0) continue;
      if((p.ny - q.ny) - Qy != 0) continue;
      if((p.nz - q.nz) - Qz != 0) continue;
      non_zero_loop_indices[ip].push_back(iq);
    }
  }
}

double OneBodyOperator::Norm() const
{
  return std::real((Matrix * Matrix.adjoint()).trace());
}

void OneBodyOperator::Erase()
{
  Matrix.setConstant({0.0,0.0});
}

double OneBodyOperator::Memory()
{
  int norb = modelspace->GetNumberSPStates();
  return sizeof(std::complex<double>) * (double)norb * (double)norb; // in unit of byte
}

void OneBodyOperator::SetSpinOperator()
{
  for(int ip : modelspace->all_states){
    SPState& p = modelspace->GetSPState(ip);
    for(int iq : GetNonZeroLoop(ip)){
      SPState& q = modelspace->GetSPState(iq);
      if(p.sz != q.sz) continue;
      double me = 0.5 * (double)p.sz;
      SetME(ip, iq, me);
    }
  }
}

bool OneBodyOperator::IsZero(double threshold) const
{
  return (std::pow(Norm(),2) < threshold) ? true : false;
}

bool OneBodyOperator::IsDiagonal(double threshold) const
{
  Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> tmp = Matrix - Matrix.diagonal();
  return (std::pow(std::real((tmp * tmp.adjoint()).trace()),2) < threshold) ? true : false;
}
