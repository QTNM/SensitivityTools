// Collect all physical and utility constants in one specific place

#ifndef TBCONSTANTS_HH
#define TBCONSTANTS_HH 1

#include <cmath>
#include <array>
#include <complex>
#include <functional>

// references for value updates:
// [C1] CODATA 2022: P.J. Mohr et al, Rev. Mod. Phys. 97 (2025) 025002
// [C2] M.M. Restrepo and E.G. Myers, PRL 131 (2023) 243002
namespace TBeta
{
  static constexpr double pi        = 3.141592653589793238462643383279502884;    // Pi
  static constexpr double twopi     = 2.0 * pi;           // 2 Pi
  static constexpr double me        = 510.99895069;       // [C1] electron mass [keV]
  static constexpr double gA        = 1.2646;             // nucleon axial coupling
  static constexpr double gAq       = 1.24983;            // quenched gA
  static constexpr double gV        = 1.0;                // nucleon vector coupling
  static constexpr double MTr       = 2808921.1367789;    // [C2] bare nuclear tritium mass [keV]
  static constexpr double Mf        = 2808391.6111557;    // [C2] bare nuclear 3He+ mass [keV]
  static constexpr double alpha     = 7.2973525643e-3;    // [C1] ine structure constant
  static constexpr double Gf        = 1.1663787e-17;      // Fermi interaction strength [keV^-2]
  static constexpr double Rn        = 2.884e-3;           // nuclear radius of 3He [me]
  static constexpr double keVInvSec = 1.52e18;            // conversion [keV s]
  static constexpr double secYear   = 60*60*24*365.25;    // conversion [s/yr]    
  static constexpr double Ryd       = 13.605693122994e-3; // Rydberg energy [keV]
  static constexpr double Vud       = 0.97425;            // CKM matrix element
  static constexpr double MAt       = MTr+me-Ryd;         // atomic tritium mass including binding energy
  
  // PMNS first row squared mixing elements
  static constexpr double s12       = 0.297;
  static constexpr double s13NO     = 0.0215;
  static constexpr double s13IO     = 0.0216;
  static constexpr std::array<double,3> UeSqNO = {(1.0-s12)*(1-s13NO), s12*(1.0-s13NO), s13NO};
  static constexpr std::array<double,3> UeSqIO = {(1.0-s12)*(1-s13IO), s12*(1.0-s13IO), s13IO};
  
  // Squared mass differences [keV]
  static constexpr double dm21Sq =  7.42e-11;
  static constexpr double dm31Sq =  2.517e-9;
  static constexpr double dm32Sq = -2.498e-9;
  
  // formula constants
  static constexpr double emin = 0.2; // [keV] low-energy cut-off, avoiding atomic effects in S(Z,en)
  static constexpr double v0   = 76.0e-3; // =76 eV in ref [1]
  static constexpr double mcc  = Mf / me; // mass of He3 in units of me [1]

  //
  // function interfaces
  //

  // heaviside function
  double heavyside(double x){
    double result;
    if (x>0.0) result = 1.0;
    else if (x==0.0) result = 0.5;
    else result = 0.0;
    return result;
  }
  
  // Lanczos approximation - needs testing!
  std::complex<double> Gamma(std::complex<double> z) {
    double g = 7;
    static const double p[9] = {0.99999999999980993, 676.5203681218851, -1259.1392167224028,
				771.32342877765313, -176.61502916214059, 12.507343278686905,
				-0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7};
    
    if (z.real()<0.5)
      return (pi / (std::sin(pi * z)*Gamma(1.0-z)));
    else {
      z -= 1;
      std::complex<double> x = p[0];
      for (int i = 1; i < 9; i++)
	x += p[i]/(z+(double)i);
      std::complex<double> t = z + g + 0.5;
      return std::sqrt(twopi) * (std::pow(t, z + 0.5)) * std::exp(-t) * x;
    }
  }
  
  // Simpson numerical integration
  class SimpsonIntegral {
  private:
    // data
    std::function<double(double,double,double)> func_;
    // function parameter
    double en_;
    double munu_;
    // internal simpson method
    inline double simpson(double val, double step) {
      // enforce integration variable as first argument
      return (func_(val,en_,munu_) + 4.0*func_(val+step/2.0,en_,munu_)
	      + func_(val+step,en_,munu_))/6.0;
    }
  public:
    // restrict constructor to fixed function signature
    SimpsonIntegral(std::function<double(double,double,double)>& f, double e, double m) {
      en_ = e;   // extract two parameters explicitly
      munu_ = m; // from function interface
      func_ = f; // function pointer
    };
    // single action member function
    inline double integrate (double from, double to, int steps) {
      double s = 0.0;
      double h = (to-from)/steps;
      for (int i = 0; i < steps; ++i)
	s += simpson(from + h*i, h);
      return h*s;
    }
  };
  
  // Collect inline functions in one specific place
  // Refs quoted:
  // [1] Preprint arXiv 1806.00369
  // [2] PRL 5(85) 807

    // Atomic 3He mass including n-th energy level binding
    double MfAt(int n){
        if (n<1) n=1; // remove illegal n values
        return Mf + me - 4.0*Ryd/(n*n);
    }

    // 3-body endpoint energy of atomic tritium [keV] (neutrino mass [keV])
    // and energy level n
    double endAt(double munu, int n){
        double mat2  = MAt*MAt;
        double me2   = me*me;
        double matn  = MfAt(n);
        double msum2 = (matn+munu)*(matn+munu);
        return (mat2 + me2 - msum2)/(2.0*MAt) - me;
    }

    // Simpson approximation of Fermi function [1] (A.2)
    double Fermi(int Z, double beta) {
        double eta   = alpha * Z / beta; // Sommerfeld parameter
        double nom   = twopi * eta * (1.002037 - 0.001427*beta);
        double denom = 1.0-std::exp(-twopi * eta);
        return nom / denom;
    }

  // Radiative correction [1] (A.3)
  double G(double en, double endp) {
    if (endp<=en || en<emin) return 0.0;
    double w  = (en + me) / me; // total electron energy [me]
    double w0 = (endp + me) / me;
    double p  = std::sqrt(w*w - 1.0);
    double beta = p/w;
    double t  = (1.0/beta)*std::atanh(beta) - 1.0;
    double fac1  = std::pow((w0-w),(2.0*alpha*t/pi));
    double fac2  = 2.0*alpha/pi;
    double term1 = t*(std::log(2.0) - 3.0/2.0 + (w0-w)/w);
    double term2 = (t+1)/4.0 * (2.0*(1.0+beta*beta) + 2.0*std::log(1.0-beta) + (w0-w)*(w0-w)/(6.0*w*w));
    return fac1*(1.0+fac2*(term1 + term2 - 2.0 + beta/2.0 - 17.0/36.0 * beta*beta + 5.0/6.0 * beta*beta*beta));
  }

  // Orbital electron shielding [1] (A.4)
  double S(int Z, double en) {
    if (en<emin) return 1.0; // check announced anomaly
    double w  = (en + me) / me; // total electron energy [me]
    double p  = std::sqrt(w*w - 1.0);
    double wb = w - v0 / me;
    double pb = std::sqrt(wb*wb - 1.0);
    double eta   = alpha * Z*w/p;
    double etab  = alpha * Z*wb/pb;
    double gamma = std::sqrt(1.0-(alpha*alpha*Z*Z));
    double fac1 = wb/w*std::pow(pb/p, -1.0+2.0*gamma);
    std::complex<double> arg1 (gamma, etab), arg2 (gamma, eta);
    std::complex<double> d1 = Gamma(arg1);
    std::complex<double> d2 = Gamma(arg2);
    double nom   = std::norm(d1 * d1);
    double denom = std::norm(d2 * d2);
    return fac1 * std::exp(pi*(etab-eta)) * nom/denom;
  }
  
  // Scaling od the electric field within nucleus [1] (A.7)
  double L(int Z, double en) {
    double w  = (en + me) / me; // total electron energy [me]
    double fac = alpha * Z;
    double gamma = std::sqrt(1.0-fac*fac);
    double term1 = w*Rn*fac/15.0 * (41.0-26.0*gamma)/(2.0*gamma-1.0);
    double term2 = fac*Rn*gamma/(30.0*w) * (17.0-2.0*gamma)/(2.0*gamma-1.0);
    return 1.0 + 13.0/60.0*fac*fac - term1 - term2;
  }
  
  // Convolution of electron and neutrino wavefunctions within nucleus [1] (A.8)
  double CC(int Z, double en, double endp) {
    if (endp<=en || en<emin) return 1.0;
    double w  = (en + me) / me; // total electron energy [me]
    double w0 = (endp + me) / me;
    double fac = alpha * Z;
    double C0 = -233.0/630.0*fac*fac - 1.0/5.0*w0*w0*Rn*Rn + 2.0/35.0*w0*Rn*fac;
    double C1 = -21.0/35.0*Rn*fac + 4.0/9.0*w0*Rn*Rn;
    double C2 = -4.0/9.0*Rn*Rn;
    return 1.0 + C0 + C1*w + C2*w*w;
  }
  

    // Recoiling nuclear charge field [1] (A.9)
  double Q(int Z, double en, double endp) {
    if (endp<=en || en<emin) return 1.0;
    double lt = gA / gV;
    double w   = (en + me) / me; // total electron energy [me]
    double w0  = (endp + me) / me;
    double p   = std::sqrt(w*w - 1.0);
    double fac = pi * alpha * Z / (mcc * p);
    return 1.0 - fac*(1.0 + (1.0-lt*lt)/(1.0+3.0*lt*lt) * (w0-w)/(3.0*w));
  }
  
  // Combined correction factor for atomic tritium
  // function of neutrino mass munu [keV], n atomic energy level
  // of daughter nucleus, en electron energy [kev]
  double Corr(double en, double munu, int n) {
    double e0 = endAt(munu, n);
    if (e0<=en || en<emin) return 1.0; // no correction
    double arg = std::sqrt((en+me)*(en+me)-me*me)/(en+me);
    return Fermi(2,arg)*S(2,en)*G(en,e0)*L(2,en)*CC(2,en,e0)*Q(2,en,e0);
  }

  // Differential decay rate with energy en [kev], applicable to LH SM currents
  // for the emission of an electron antineutrino with mass munu [kev] and 
  // the endpoint of the n-th 3He energy level. With corrections.
  double stub(double en, double munu, double e0) { // used twice, factor out
    if (e0<en || en<emin) return 0.0;
    double fac1  = (Gf*Gf*Vud*Vud)/(2.0*pi*pi*pi);
    double denom = MTr*MTr - 2.0*MTr*(en+me) + me*me;
    double nom1  = MTr*(en+me) - me*me;
    double nom2  = (en+me)*(en+me) - me*me;
    double fac2  = MTr*MTr*std::sqrt(nom2)/denom;
    double fac3  = std::sqrt((e0-en)*(e0-en+2.0*munu*Mf/MTr));
    double fac4  = (gV+gAq)*(gV+gAq);
    double fac5  = MTr*(MTr-en-me)/denom;
    double fac6  = (e0-en+(munu*(munu+Mf)/MTr))*nom1/denom;
    double fac7  = e0-en+(Mf*(munu+Mf)/MTr);
    double term = -1.0/3.0*(MTr*MTr*nom2/(denom*denom) * (e0-en)*(e0-en+2.0*munu*Mf/MTr));
    double term1 = fac2*fac3*(fac4*fac5*fac6*fac7+term);
    double fac8  = (gV-gAq)*(gV-gAq);
    double term2 = fac8*(en+me)*(e0-en+munu*Mf/MTr);
    double fac9  = gAq*gAq - gV*gV;
    double term3 = fac9*Mf*fac6;
    return fac1*(term1+term2+term3);
  }

    double Diff(double en, double munu, int n) {
        double e0 = endAt(munu, n);
        double fac = stub(en, munu, e0);
        return fac * Corr(en,munu,n);
    }

  // neutrino mass spectrum
  // order is boolean True for Normal order (NO)
  // False for Inverted order (IO)
  const std::array<double,3>& nuSpectrum(bool order, double munu) {
    static const std::array<double,3> no = {munu, std::sqrt(munu*munu+dm21Sq), std::sqrt(munu*munu+dm31Sq)};
    static const std::array<double,3> io = {std::sqrt(munu*munu-dm21Sq-dm32Sq), std::sqrt(munu*munu-dm32Sq), munu};
    return order ? no : io;
  }

  // like Diff but summing over all three light neutrinos with weights.
  // order is boolean True for Normal order (NO)
  // False for Inverted order (IO)
  double Diff3nu(bool order, double en, double munu, int n) {
    const std::array<double,3>& UeSq = order ? UeSqNO : UeSqIO;
    double  sum  = 0.0;
    for (int i=0;i<3;++i) {
      sum += UeSq[i] * Diff(en, nuSpectrum(order, munu)[i], n);
    }
    return sum;
  }
  
    // like Diff but summing over all three light neutrinos with weights and
    // a sterile neutrino with mass mN [keV] with active-sterile mixing
    // strength eta (0<=eta<1)
    double Diff4nu(bool order, double mN, double eta, double en, double munu, int n) {
        double e0 = endAt(mN, n);
        return (1.0-eta*eta)*Diff3nu(order, en, munu, n) + eta*eta*Diff(en, mN, n)*heavyside(e0-en);
    }

    // Sum over discrete atomic energy levels of 3He
    // with branching ratios, [2]
    double etaL(double en) {
        double denom  = (en+me)*(en+me) - me*me;
        return -2.0*alpha*me/std::sqrt(denom);
    }

    double aL(double en) {
        double eta = etaL(en);
        double nom = std::exp(2.0*eta*std::atan(-2.0/eta));
        double denom = (1.0 + eta*eta/4.0)*(1.0 + eta*eta/4.0);
        return eta*eta*eta*eta*nom/denom;
    }

    double Lev(int n, double en) {
        double al = aL(en);
        if (n==2) // special case
            return 0.25 * (1.0 + al*al - al);
        double term1 = 256.0*std::pow(n, 5)*std::pow(n-2, 2*n-4)/std::pow(n+2, 2*n+4);
        double term2 = al*al/(n*n*n) - 16.0*n*al*std::pow(n-2, n-2)/std::pow(n+2, n+2);
        return 2.0*(term1+term2);
    }

    // Full (SM+sterile) differential decay rate as function of electron energy [keV].
    // Includes all corrections (no continuum orbital e- states) and 
    // the first 5 bound discrete atomic states.
    double dGammadE(bool order, double munu, double mN, double eta, double en) {
        double sum = 0.0;
        for (int n=1;n<=5;++n)
            sum += Lev(n, en) * Diff4nu(order, mN, eta, en, munu, n);
        return sum;
    }

    // Set up all contributions including continous states and first 5 
    // discrete states. Here consider state n as double for continuum states.
    double endCont(double munu, double n) {
        return endAt(munu, 1)-4.0*Ryd*(1.0+1.0/(n*n));
    }

    double CorrCont(double en, double munu, double n) {
        double e0 = endCont(munu, n);
        double nom  = (en+me)*(en+me) - me*me;
        double fac = Fermi(2,std::sqrt(nom)/(en+me))*S(2,en)*G(en,e0);
        return fac*L(2,en)*CC(2,en,e0)*Q(2,en,e0);
    }

    double DiffCont(double en, double munu, double n) {
        double e0 = endCont(munu, n);
        double fac = stub(en, munu, e0);
        return fac * CorrCont(en, munu, n);
    }

    // integrand over states g
    double integrand(double g, double en, double munu) {
        double fac1 = DiffCont(en, munu, g)*twopi/(g*g*g*(std::exp(twopi*g)-1.0));
        double fac2 = g*g*g*g*std::exp(2.0*g*std::atan(-2.0/g))/((1.0+g*g/4.0)*(1.0+g*g/4.0));
        return fac1*(fac2*fac2 + aL(en)*aL(en) - aL(en)*fac2);
    }

    // Contribution only from the continuum orbital electron states.
    // integrate over g.
    double gammaCont(double en, double munu) {
        if (en > endCont(munu, 1e10) || en<emin) return 0.0;
        std::function<double(double,double,double)> fn = integrand;
        SimpsonIntegral sint(fn, en, munu);
        return sint.integrate(-99, etaL(en), 1000)/pi;
    }

  double dGammadECont(bool order, double munu, double mN, double eta, double en) {
    const std::array<double,3>& UeSq = order ? UeSqNO : UeSqIO;
    double sum = 0.0;
    for (int i=0;i<3;++i) {
      sum += UeSq[i] * gammaCont(en, nuSpectrum(order,munu)[i]);
        }
    return (1-eta*eta)*sum + eta*eta*gammaCont(en, mN);
  }

  double dGammadEFull(bool order, double munu, double mN, double eta, double en) {
    return dGammadE(order, munu, mN, eta, en) + dGammadECont(order, munu, mN, eta, en);
  }
} // namespace TBeta

#endif
