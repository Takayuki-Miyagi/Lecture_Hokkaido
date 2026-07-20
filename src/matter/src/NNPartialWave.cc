#include "Constants.hh"
#include "NNPartialWave.hh"

NNPartialWave::~NNPartialWave()
{}

NNPartialWave::NNPartialWave():
  Weighted(false)
{}

NNPartialWave::NNPartialWave(std::string filename):
  filename(filename), Weighted(false)
{
  if ( not std::ifstream(filename).good() )
  {
    std::cout << std::endl << "========================================" << std::endl;
    std::cout <<  __func__ << "  No such file : " << filename;
    std::cout << std::endl << "========================================" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::ifstream infile(filename, std::ios::binary);
  infile.read(reinterpret_cast<char*>(&NMesh), sizeof(NMesh));
  infile.read(reinterpret_cast<char*>(&Jmax), sizeof(Jmax));
  infile.read(reinterpret_cast<char*>(&NChan), sizeof(NChan));
  Mesh.resize(NMesh);
  Weight.resize(NMesh);
  infile.read(reinterpret_cast<char*>(Mesh.data()), Mesh.size()*sizeof(double));
  infile.read(reinterpret_cast<char*>(Weight.data()), Weight.size()*sizeof(double));
  std::array<int,5> tmp;
  for(int ichan=0; ichan<NChan; ichan++){
    infile.read(reinterpret_cast<char*>(&tmp), 5*sizeof(int)); // J, Parity, S, Tz, Ndim
    int J, Parity, S, Tz, Ndim, L, Lp;
    J = tmp[0]; Parity = tmp[1]; S = tmp[2]; Tz = tmp[3]; Ndim = tmp[4];
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> mat;
    mat.resize(Ndim, Ndim);
    infile.read(reinterpret_cast<char*>(mat.data()), mat.size()*sizeof(double));

    if(Ndim==NMesh){
      if(std::pow(-1, J)==Parity){
        L = J; Lp = J;
        JLLSTz.push_back({J,L,Lp,S,Tz});
        ChannelIndex[{J,L,Lp,S,Tz}] = JLLSTz.size()-1;
        Vmat.push_back(mat);
      }
      else{
        L = J+1; Lp = J+1;
        JLLSTz.push_back({J,L,Lp,S,Tz});
        ChannelIndex[{J,L,Lp,S,Tz}] = JLLSTz.size()-1;
        Vmat.push_back(mat);
      }
    }
    else{
        L = J-1; Lp = J-1;
        JLLSTz.push_back({J,L,Lp,S,Tz});
        ChannelIndex[{J,L,Lp,S,Tz}] = JLLSTz.size()-1;
        Vmat.push_back(mat(Eigen::seq(0,NMesh-1), Eigen::seq(0,NMesh-1)));

        L = J+1; Lp = J-1;
        JLLSTz.push_back({J,L,Lp,S,Tz});
        ChannelIndex[{J,L,Lp,S,Tz}] = JLLSTz.size()-1;
        Vmat.push_back(mat(Eigen::seq(NMesh,Ndim-1), Eigen::seq(0,NMesh-1)));

        L = J-1; Lp = J+1;
        JLLSTz.push_back({J,L,Lp,S,Tz});
        ChannelIndex[{J,L,Lp,S,Tz}] = JLLSTz.size()-1;
        Vmat.push_back(mat(Eigen::seq(0,NMesh-1), Eigen::seq(NMesh,Ndim-1)));

        L = J+1; Lp = J+1;
        JLLSTz.push_back({J,L,Lp,S,Tz});
        ChannelIndex[{J,L,Lp,S,Tz}] = JLLSTz.size()-1;
        Vmat.push_back(mat(Eigen::seq(NMesh,Ndim-1), Eigen::seq(NMesh,Ndim-1)));
    }
  }
  SolveDeuteron(); // test function
}

void NNPartialWave::SolveDeuteron()
{
  WeightNormalize();
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Hmat;
  Hmat.resize(2*NMesh, 2*NMesh);
  Hmat.setConstant(0.0);
  double mred = Constants::M_PROTON * Constants::M_NEUTRON / (Constants::M_PROTON + Constants::M_NEUTRON);
  for(int i=0; i<NMesh; i++){
    double mom = Mesh(i);
    Hmat(i,i) = std::pow(mom * Constants::HBARC, 2) / (2*mred);
    Hmat(i+NMesh,i+NMesh) = std::pow(mom * Constants::HBARC, 2) / (2*mred);
  }

  Hmat(Eigen::seq(0,NMesh-1), Eigen::seq(0,NMesh-1)) += Vmat[ChannelIndex[{1,0,0,1,0}]];
  Hmat(Eigen::seq(0,NMesh-1), Eigen::seq(NMesh,2*NMesh-1)) += Vmat[ChannelIndex[{1,0,2,1,0}]];
  Hmat(Eigen::seq(NMesh,2*NMesh-1), Eigen::seq(0,NMesh-1)) += Vmat[ChannelIndex[{1,2,0,1,0}]];
  Hmat(Eigen::seq(NMesh,2*NMesh-1), Eigen::seq(NMesh,2*NMesh-1)) += Vmat[ChannelIndex[{1,2,2,1,0}]];

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> eigensolver(Hmat);
  if (eigensolver.info() != Eigen::Success) abort();
  std::cout << " Eigen energies [NN, 1+ (deuteron channel)]: " << eigensolver.eigenvalues()(Eigen::seq(0,4)).transpose().format(6) << std::endl;
  WeightUnNormalize();
}

void NNPartialWave::WeightNormalize()
{
  for(auto &it : ChannelIndex){
    auto& mat = Vmat[it.second];
    for(int i=0; i<NMesh; i++){
      for(int j=0; j<NMesh; j++){
        mat(i,j) *= Mesh(i) * Mesh(j) * std::sqrt(Weight(i) * Weight(j));
      }
    }
  }
}

void NNPartialWave::WeightUnNormalize()
{
  for(auto &it : ChannelIndex){
    auto& mat = Vmat[it.second];
    for(int i=0; i<NMesh; i++){
      for(int j=0; j<NMesh; j++){
        mat(i,j) /= Mesh(i) * Mesh(j) * std::sqrt(Weight(i) * Weight(j));
      }
    }
  }
}

void NNPartialWave::FixPhase()
{
  for(auto &it : ChannelIndex){
    int L, Lp;
    L = it.first[1];
    Lp= it.first[2];
    auto& mat = Vmat[it.second];
    mat *= std::pow(-1.0, (L-Lp)/2);
  }
}
