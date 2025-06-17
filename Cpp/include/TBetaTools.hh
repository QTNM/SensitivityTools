// Collect all Tritium Beta Decay tools, using the generator header

#ifndef TBTOOLS_HH
#define TBROOLS_HH 1

#include <cmath>
#include "TBetaGenerator.hh"

namespace TBTools
{
  //
  // function interfaces
  //

  // Kurie transformation
  inline double KurieTrsf(double en, double y){
    double me = TBeta::me;
    double momentum = std::sqrt((en + me)*(en + me) - (me*me));
    double arg = momentum / (en+me);
    double fm = TBeta::Fermi(2, arg);
    return std::sqrt(y / ((en + me)*momentum*fm));
  }

} // namespace TBTools

#endif
