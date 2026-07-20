#ifdef __APPLE__
#include <mach/mach.h>
#endif
#include "Profiler.hh"
#include <algorithm>
#include <sys/time.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/resource.h>
#include <omp.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

namespace Profiler{
  std::map<std::string, double> timer;
  std::map<std::string, int> counter;
  double start_time = -1;

  void Start()
  {
    if (start_time < 0)
    {
      start_time = omp_get_wtime();
      counter["N_Threads"] = omp_get_max_threads();
    }
  }

  std::map<std::string,size_t> CheckMem()
  {
    std::map<std::string,size_t> s;
    s["RSS"]=0;
#ifdef __APPLE__
    task_basic_info_data_t info;
    mach_msg_type_number_t size = TASK_BASIC_INFO_COUNT;
    task_info(mach_task_self(), TASK_BASIC_INFO, (task_info_t)&info, &size);
    s["RSS"] = info.resident_size / 1024.0;
#else
    std::ifstream ifs("/proc/self/statm"); // this is a pseudo file that provides information on the current process (linux only)
    if ( ifs.good() )
    {
      size_t npages;
      size_t page_size = sysconf(_SC_PAGESIZE);
      ifs >> npages >> npages;
      s["RSS"] = npages * page_size /1024.0; // divide by 1024 to convert to kilobytes
    }
#endif
    return s;
  }

  size_t MaxMemUsage()
  {
    struct rusage usage;
    getrusage(RUSAGE_SELF,&usage);
#ifdef __APPLE__
    return usage.ru_maxrss / 1024.0;  // convert bytes to KB
#else
    return usage.ru_maxrss;         // already in KB
#endif
  }

  std::map<std::string,float> GetTimes()
  {
    struct rusage ru;
    getrusage(RUSAGE_SELF,&ru);
    std::map<std::string,float> times;
    times["user cpu time"] = ru.ru_utime.tv_sec + 1e-6*ru.ru_utime.tv_usec;
    times["system cpu time"] = ru.ru_stime.tv_sec + 1e-6*ru.ru_stime.tv_usec;
    times["walltime"] = omp_get_wtime() - start_time;
    return times;
  }

  void PrintTimes()
  {
    auto time_tot = GetTimes();
    std::cout << "====================== TIMES (s) ====================" << std::endl;
    std::cout.setf(std::ios::fixed);
    size_t max_func_length = 40;
    for ( auto it : timer ) {
      max_func_length = std::max( it.first.size(), max_func_length);
    }
    max_func_length += 5;
    for ( auto it : timer )
    {
      int nfill = (int) (20 * std::min( 1.0, it.second / time_tot["walltime"]));
      std::cout << std::fixed << std::setprecision(2) << std::setw(max_func_length) << std::left << it.first + ":  " << std::setw(12) << std::setprecision(5) << std::right << it.second;
      std::cout << " (" << std::setw(4) << std::fixed << std::setprecision(1) << 100*it.second / time_tot["walltime"] << "%) |";
      for (int ifill=0; ifill<nfill; ifill++) std::cout << "*";
      for (int ifill=nfill; ifill<20; ifill++) std::cout << " ";
      std::cout << "|";
      std::cout << std::endl;

    }
    std::cout << std::setw(40) << std::left << "      walltime:" << std::setw(12) << std::setprecision(5) << std::right << time_tot["walltime"]  << std::endl;
    std::cout << std::setw(40) << std::left << "system cpu time:" << std::setw(12) << std::setprecision(5) << std::right << time_tot["system cpu time"]  << std::endl;
    std::cout << std::setw(40) << std::left << "  user cpu time:" << std::setw(12) << std::setprecision(5) << std::right << time_tot["user cpu time"]  << std::endl;
  }

  void PrintCounters()
  {
    std::cout << "===================== COUNTERS =====================" << std::endl;
    std::cout.setf(std::ios::fixed);
    for ( auto it : counter )
      std::cout << std::setw(40) << std::left << it.first + ":  " << std::setw(12) << std::setprecision(0) << std::right << it.second  << std::endl;
  }

  void PrintMemory()
  {
    std::cout << "===================== MEMORY (GB) ==================" << std::endl;
    for (auto it : CheckMem())
      std::cout << std::fixed << std::setw(40) << std::left << it.first + ":  " << std::setw(12) << std::setprecision(3) << std::right << it.second/1024.0/1024.0 << std::endl;

    std::cout << std::fixed << std::setw(40) << std::left << "Max Used:  " << std::setw(12) << std::setprecision(3) << std::right << MaxMemUsage()/1024.0/1024.0  << std::endl;
  }

  void PrintAll()
  {
    PrintCounters();
    PrintTimes();
    PrintMemory();
  }
}
