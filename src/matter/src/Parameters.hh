#ifndef Parameters_h
#define Parameters_h 1

#include <map>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <iomanip>

class Parameters
{
  public:
    static std::map<std::string,std::string> strings;
    static std::map<std::string,double> doubles;
    static std::map<std::string,int> integers;
    static std::map<std::string,std::vector<std::string>> vecs;

    Parameters(){};
    Parameters(int, char**);
    void ParseCommandLineArgs(int, char**);
    std::string s(std::string);
    double d(std::string);
    int i(std::string);
    std::vector<std::string> v(std::string);
    void PrintOptions();
};

std::map<std::string,std::string> Parameters::strings = {
  {"mode",  "snm"},   // pure neutron matter (pnm), symmetric nuclear matter (snm), or none. If you set none, Zu, Zd, Nu, Nd inputs will be used.
  {"chiral_delta", "true"}, // delta-full or delta-less chiral EFT
  {"include_3NF", "true"}, // include 3NF or not
  {"method", "magnus"}, // HF, MBPT, magnus, RK4
  {"truncation_method", "box"}, // truncation method either box or shell
  {"input_nn_file", "none"}, // input NN PW file
};


std::map<std::string,double> Parameters::doubles = {
  {"rho", 0.16}, // density of nuclear matter in unit of fm-3
  {"Ct1S0_pp", 0}, // LO Contact LEC pp in unit of 10^4 GeV-2
  {"Ct1S0_np", 0}, // LO Contact LEC np in unit of 10^4 GeV-2
  {"Ct1S0_nn", 0}, // LO Contact LEC nn in unit of 10^4 GeV-2
  {"Ct3S1",    0}, // LO Contact LEC in unit of 10^4 GeV-2
  {"C1S0",     0}, // NLO Contact LEC in unit of 10^4 GeV-4
  {"C3S1",     0}, // NLO Contact LEC in unit of 10^4 GeV-4
  {"C1P1",     0}, // NLO Contact LEC in unit of 10^4 GeV-4
  {"C3P0",     0}, // NLO Contact LEC in unit of 10^4 GeV-4
  {"C3P1",     0}, // NLO Contact LEC in unit of 10^4 GeV-4
  {"C3SD1",    0}, // NLO Contact LEC in unit of 10^4 GeV-4
  {"C3P2",     0}, // NLO Contact LEC in unit of 10^4 GeV-4
  {"c1",       0}, // N2LO 2pi LEC in unit of GeV-1
  {"c2",       0}, // N2LO 2pi LEC in unit of GeV-1
  {"c3",       0}, // N2LO 2pi LEC in unit of GeV-1
  {"c4",       0}, // N2LO 2pi LEC in unit of GeV-1
  {"cD",       0}, // N2LO 3N LEC in dimensionless unit
  {"cE",       0}, // N2LO 3N LEC in dimensionless unit
  {"RegLambda",450}, // Regulator cutoff momentum in unit of MeV
  {"SFRLambda",700}, // SFR momentum in unit of MeV
  {"smax",100}, // IMSRG smax
  {"ds_max", 0.2}, // IMSRG ds max
  {"omega_norm_max", 0.2}, // IMSRG max(||Omega||)
  {"eta_criterion", 1.e-6}, // IMSRG max(||Omega||)
};

std::map<std::string,int> Parameters::integers = {
  {"A",	132},	// mass number
  {"Zu", -1},	// up proton number
  {"Zd",-1},	// down proton number
  {"Nu",-1},	// up neutron number
  {"Nd",-1},	// down neutron number
  {"Nmax", 2},	// size of model space
  {"ch_order", 0},	// chiral order
  {"reg_power", 3},	// regulator power
  {"n_TABC", -1},	// number for twist-averaged boundary condition
};

std::map<std::string,std::vector<std::string>> Parameters::vecs = {
  {"",{} },
};


// The constructor
Parameters::Parameters(int argc, char** argv)
{
  ParseCommandLineArgs(argc, argv);
}

void Parameters::ParseCommandLineArgs(int argc, char** argv)
{
  std::cout << "====================  Parameters (defaults set in Parameters.hh) ===================" << std::endl;
  for (int iarg=1; iarg<argc; ++iarg)
  {
    std::string arg = argv[iarg];
    size_t pos = arg.find("=");
    std::string var = arg.substr(0,pos);
    std::string val = arg.substr(pos+1);
    if (strings.find(var) != strings.end() )
    {
      if (val.size() > 0)
        std::istringstream(val) >> strings[var];
      std::cout << var << " => " << strings[var] << std::endl;
    }
    else if (doubles.find(var) != doubles.end() )
    {
      if (val.size() > 0)
        std::istringstream(val) >> doubles[var];
      std::cout << var << " => " << doubles[var] << std::endl;
    }
    else if (integers.find(var) != integers.end() )
    {
      if (val.size() > 0)
        std::istringstream(val) >> integers[var];
      std::cout << var << " => " << integers[var] << std::endl;
    }
    else if (vecs.find(var) != vecs.end() )
    {
      if (val.size() > 0)
      {
        std::istringstream ss(val);
        std::string tmp;
        while( getline(ss,tmp,',')) vecs[var].push_back(tmp);
      }
      std::cout << var << " => ";
      for (auto x : vecs[var]) std::cout << x << ",";
      std::cout << std::endl;
    }
    else
    {
      std::cout << "Unkown parameter: " << var << " => " << val << std::endl;
    }

  }
  std::cout << "====================================================================================" << std::endl;
}

std::string Parameters::s(std::string key)
{
  return strings[key];
}
double Parameters::d(std::string key)
{
  return doubles[key];
}
int Parameters::i(std::string key)
{
  return integers[key];
}
std::vector<std::string> Parameters::v(std::string key)
{
  return vecs[key];
}

void Parameters::PrintOptions()
{
  std::cout << "Input parameters and default values: " << std::endl;
  for (auto& strpar : strings)
  {
    std::cout << "\t" << std::left << std::setw(30) << strpar.first << ":  " << strpar.second << std::endl;
  }
  for (auto& doublepar : doubles)
  {
    std::cout <<  "\t" << std::left << std::setw(30) <<doublepar.first << ":  " << doublepar.second << std::endl;
  }
  for (auto& intpar : integers)
  {
    std::cout <<  "\t" << std::left << std::setw(30) <<intpar.first << ":  " << intpar.second << std::endl;
  }
  for (auto& vecpar : vecs)
  {
    std::cout <<  "\t" << std::left << std::setw(30) << vecpar.first << ":  ";
    for (auto& op : vecpar.second) std::cout << op << ",";
    std::cout << std::endl;
  }

}
#endif
