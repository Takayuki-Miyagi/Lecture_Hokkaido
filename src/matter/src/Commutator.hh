#ifndef Commutator_h
#define Commutator_h 1

#include "Profiler.hh"
#include "Operator.hh"

namespace Commutator
{
  inline bool use_naive_implementation = false;
  inline void naive_implementation(bool slow) {use_naive_implementation=slow;};
  Operator Commutator(const Operator& X, const Operator&Y);
  void comm110(const Operator& X, const Operator& Y, Operator& Z);
  void comm220(const Operator& X, const Operator& Y, Operator& Z);
  void comm111(const Operator& X, const Operator& Y, Operator& Z);
  void comm121(const Operator& X, const Operator& Y, Operator& Z);
  void comm221(const Operator& X, const Operator& Y, Operator& Z);
  void comm122(const Operator& X, const Operator& Y, Operator& Z);
  void comm222(const Operator& X, const Operator& Y, Operator& Z);
};
#endif

