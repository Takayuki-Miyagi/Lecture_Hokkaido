#ifndef Profiler_h
#define Profiler_h 1
#include <map>
#include <string>

namespace Profiler
{
  extern std::map<std::string, double> timer;
  extern std::map<std::string, int> counter;
  extern double start_time;

  void Start();
  std::map<std::string,size_t> CheckMem();
  std::map<std::string,float> GetTimes();
  void PrintTimes();
  void PrintCounters();
  void PrintMemory();
  void PrintAll();
  size_t MaxMemUsage();
};

#endif

