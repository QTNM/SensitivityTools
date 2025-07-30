''' Translation of TBetaDecayGenerator.hpp '''
import numpy as np
from scipy.integrate import quad
from scipy.special import gamma
from numba import njit

# references for value updates:
# [C1] CODATA 2022: P.J. Mohr et al, Rev. Mod. Phys. 97 (2025) 025002
# [C2] M.M. Restrepo and E.G. Myers, PRL 131 (2023) 243002
# define constants
pi        = 3.141592653589793238462643383279502884  # Pi
twopi     = 2.0 * pi           # 2 Pi
me        = 510.99895069       # [C1] electron mass [keV]
gA        = 1.2646             # nucleon axial coupling
gAq       = 1.24983            # quenched gA
gV        = 1.0                # nucleon vector coupling
MTr       = 2808921.1367789    # [C2] bare nuclear tritium mass [keV]
Mf        = 2808391.6111557    # [C2] bare nuclear 3He+ mass [keV]
alpha     = 7.2973525643e-3    # [C1] fine structure constant
Gf        = 1.1663787e-17      # Fermi interaction strength [keV^-2]
Rn        = 2.884e-3           # nuclear radius of 3He [me]
keVInvSec = 1.52e18            # conversion [keV s]
secYear   = 60*60*24*365.25    # conversion [s/yr]    
Ryd       = 13.605693122994e-3 # Rydberg energy [keV]
Vud       = 0.97425            # CKM matrix element
MAt       = MTr+me-Ryd         # atomic tritium mass including binding energy

# PMNS first row squared mixing elements
s12       = 0.297
s13NO     = 0.0215
s13IO     = 0.0216
UeSqNO = np.array([(1.0-s12)*(1-s13NO), s12*(1.0-s13NO), s13NO])
UeSqIO = np.array([(1.0-s12)*(1-s13IO), s12*(1.0-s13IO), s13IO])

# Squared mass differences [keV]
dm21Sq =  7.42e-11
dm31Sq =  2.517e-9
dm32Sq = -2.498e-9

# formula constants
v0 = 76.0e-3 # =76 eV in ref [1]
emin = v0  # [keV] low energy cut-off, avoiding atomic effects in S(Z,en)
mcc = Mf / me  # mass of He3 in units of me [1]

#
# functions
#
@njit
def heavyside(x: np.ndarray) -> np.ndarray:
    result = np.zeros(x.shape)
    msk1 = x>0.0
    msk2 = x==0.0
    result[msk1] = 1.0
    result[msk2] = 0.5
    return result

# Refs quoted:
# [1] Preprint arXiv 1806.00369
# [2] PRL 5(85) 807

# Atomic 3He mass including n-th energy level binding
@njit
def MfAt(n: int) -> float:
    if n<1: 
        n=1  # remove illegal n values
    return Mf + me - 4.0*Ryd/(n*n)

# 3-body endpoint energy of atomic tritium [keV] (neutrino mass [keV])
# and energy level n
@njit
def endAt(munu: float, n: int) -> float:
    mat2  = MAt*MAt
    me2   = me*me
    matn  = MfAt(n)
    msum2 = (matn+munu)*(matn+munu)
    return (mat2 + me2 - msum2)/(2.0*MAt) - me

# Simpson approximation of Fermi function [1] (A.2)
@njit
def Fermi(Z: int, beta: np.ndarray) -> np.ndarray:
    eta   = alpha * Z / beta  # Sommerfeld parameter
    nom   = twopi * eta * (1.002037 - 0.001427*beta)
    denom = 1.0-np.exp(-twopi * eta)
    return nom / denom

# Radiative correction [1] (A.3)
@njit
def G(enin: np.ndarray, endp: float) -> np.ndarray:
    msk = np.logical_and(enin<endp, enin>=emin)
    res = np.zeros(enin.shape)  # =0 for (not msk)
    en = np.copy(enin[msk])  # shorter than en array

    w  = (en + me) / me  # total electron energy [me]
    w0 = (endp + me) / me
    p  = np.sqrt(w*w - 1.0)
    beta = p/w
    t  = (1.0/beta)*np.arctanh(beta) - 1.0
    fac1  = np.power((w0-w),(2.0*alpha*t/pi))
    fac2  = 2.0*alpha/pi
    term1 = t*(np.log(2.0) - 3.0/2.0 + (w0-w)/w)
    term2 = (t+1)/4.0 * (2.0*(1.0+beta*beta) + 
                         2.0*np.log(1.0-beta) + 
                         (w0-w)*(w0-w)/(6.0*w*w))

    res[msk] = fac1*(1.0+fac2*(term1 + term2 - 2.0 + beta/2.0 - 
                           17.0/36.0 * beta*beta + 
                           5.0/6.0 * beta*beta*beta))
    return res

# Orbital electron shielding [1] (A.4)
def S(Z: int, enin: np.ndarray) -> np.ndarray:
    msk = enin>emin
    res = np.ones(enin.shape)  # =1 for (not msk)
    en = np.copy(enin[msk])  # shorter than en array
    w  = (en + me) / me  # total electron energy [me]
    p  = np.sqrt(w*w - 1.0)
    wb = w - v0 / me
    pb = np.sqrt(wb*wb - 1.0) # issue for en<v0
    eta   = alpha * Z*w/p
    etab  = alpha * Z*wb/pb
    gam   = np.sqrt(1.0 - (alpha*alpha*Z*Z))
    fac1  = wb/w * np.power(pb/p, -1.0+2.0*gam)
    arg1  = np.vectorize(complex)(gam, etab)
    arg2  = np.vectorize(complex)(gam, eta)
    d1 = gamma(arg1)
    d2 = gamma(arg2)
    nom   = abs(d1*d1)
    denom = abs(d2*d2)
    res[msk] = fac1 * np.exp(pi*(etab-eta)) * nom/denom
    return res

# Scaling od the electric field within nucleus [1] (A.7)
@njit
def L(Z: int, en: np.ndarray) -> np.ndarray:
    w  = (en + me) / me  # total electron energy [me]
    fac = alpha * Z
    gam   = np.sqrt(1.0-fac*fac)
    term1 = w*Rn*fac/15.0 * (41.0-26.0*gam)/(2.0*gam-1.0)
    term2 = fac*Rn*gam/(30.0*w) * (17.0-2.0*gam)/(2.0*gam-1.0)
    return 1.0 + 13.0/60.0*fac*fac - term1 - term2
  
# Convolution of electron and neutrino wavefunctions within nucleus [1] (A.8)
@njit
def CC(Z: int, enin: np.ndarray, endp: float) -> np.ndarray:
    msk = np.logical_and(enin<endp, enin>=emin)
    res = np.ones(enin.shape)  # =1 for (not msk)
    en = np.copy(enin[msk])  # shorter than en array

    w  = (en + me) / me  # total electron energy [me]
    w0 = (endp + me) / me
    fac = alpha * Z
    C0 = -233.0/630.0*fac*fac - 1.0/5.0*w0*w0*Rn*Rn + 2.0/35.0*w0*Rn*fac
    C1 = -21.0/35.0*Rn*fac + 4.0/9.0*w0*Rn*Rn
    C2 = -4.0/9.0*Rn*Rn
    res[msk] = 1.0 + C0 + C1*w + C2*w*w
    return res

# Recoiling nuclear charge field [1] (A.9)
@njit
def Q(Z: int, enin: np.ndarray, endp: float) -> np.ndarray:
    msk = np.logical_and(enin<endp, enin>=emin)
    res = np.ones(enin.shape)  # =1 for (not msk)
    en = np.copy(enin[msk])  # shorter than en array

    lt = gA/gV
    w   = (en + me) / me  # total electron energy [me]
    w0  = (endp + me) / me
    p   = np.sqrt(w*w - 1.0)
    fac = pi * alpha * Z / (mcc * p)
    res[msk] = 1.0 - fac*(1.0 + (1.0-lt*lt)/(1.0+3.0*lt*lt) * (w0-w)/(3.0*w))
    return res
  
# Combined correction factor for atomic tritium
# function of neutrino mass munu [keV], n atomic energy level
# of daughter nucleus, en electron energy [kev]
def Corr(enin: np.ndarray, munu: float, n: int) -> np.ndarray:
    e0 = endAt(munu, n)
    msk = np.logical_and(enin<e0, enin>=emin)
    rarr = np.ones(enin.shape)  # =1 for (not mask)
    en = np.copy(enin[msk])
    if en.size<1:
        return rarr
    arg = np.sqrt((en+me)*(en+me)-me*me)/(en+me)
    rarr[msk] = Fermi(2,arg)*S(2,en)*G(en,e0)*L(2,en)*CC(2,en,e0)*Q(2,en,e0)
    return rarr


# Differential decay rate with energy en [kev], applicable to LH SM currents
# for the emission of an electron antineutrino with mass munu [kev] and 
# the endpoint of the n-th 3He energy level. With corrections.
@njit
def stub(enin: np.ndarray, munu: float, e0: float) -> np.ndarray:
    msk = np.logical_and(enin<e0, enin>=emin)
    res = np.zeros(enin.shape)  # =0 for (not msk)
    en = np.copy(enin[msk])  # shorter than en array
    if en.size<1:
        return res

    fac1  = (Gf*Gf*Vud*Vud) / (2.0*pi*pi*pi)
    denom = MTr*MTr - 2.0*MTr*(en+me) + me*me
    nom1  = MTr*(en+me) - me*me
    nom2  = (en+me)*(en+me) - me*me
    fac2  = MTr*MTr*np.sqrt(nom2) / denom
    fac3  = np.sqrt((e0-en) * (e0-en + 2.0*munu*Mf/MTr))
    fac4  = (gV+gAq) * (gV+gAq)
    fac5  = MTr*(MTr-en-me) / denom
    fac6  = (e0 - en + (munu*(munu+Mf)/MTr))*nom1 / denom
    fac7  = e0 - en + (Mf*(munu+Mf)/MTr)
    term = -1.0/3.0 * (MTr*MTr*nom2/(denom*denom) * 
                        (e0-en) * (e0 - en + 2.0*munu*Mf / MTr))
    term1 = fac2*fac3*(fac4*fac5*fac6*fac7 + term)
    fac8  = (gV-gAq) * (gV-gAq)
    term2 = fac8*(en+me)*(e0 - en + munu*Mf / MTr)
    fac9  = gAq*gAq - gV*gV
    term3 = fac9*Mf*fac6
    res[msk] = fac1*(term1 + term2 + term3) # remainder entries=0
    return res

def Diff(en: np.ndarray, munu: float, n: int) -> np.ndarray:
    e0 = endAt(munu, n)
    # print('in Diff: inputs ',munu,', ',n,', e0=',e0)
    fac = stub(en, munu, e0)
    return fac * Corr(en, munu, n)

# neutrino mass spectrum
# order is boolean True for Normal order (NO)
# False for Inverted order (IO)
@njit
def nuSpectrum(order: bool, munu: float):
    no = np.array([munu, np.sqrt(munu*munu+dm21Sq), np.sqrt(munu*munu+dm31Sq)])
    io = np.array([np.sqrt(munu*munu-dm21Sq-dm32Sq), np.sqrt(munu*munu-dm32Sq), munu])
    if order:
        return no
    else:
        return io

# like Diff but summing over all three light neutrinos with weights.
# order is boolean True for Normal order (NO)
# False for Inverted order (IO)
def Diff3nu(order: bool, en: np.ndarray, munu: float, n: int) -> np.ndarray:
    if order:
        UeSq = UeSqNO
    else:
        UeSq = UeSqIO
    sum  = np.zeros(en.shape)
    for i in range(3):
        sum += UeSq[i] * Diff(en, nuSpectrum(order, munu)[i], n)
    return sum

# like Diff but summing over all three light neutrinos with weights and
# a sterile neutrino with mass mN [keV] with active-sterile mixing
# strength eta (0<=eta<1)
def Diff4nu(order: bool, mN: float, eta: float, en: np.ndarray, munu: float, n: int) -> np.ndarray:
    e0 = endAt(mN, n)
    result = (1.0-eta*eta)*Diff3nu(order, en, munu, n)
    result += eta*eta*Diff(en, mN, n)*heavyside(e0-en)
    return result
    
# Sum over discrete atomic energy levels of 3He
# with branching ratios, [2]
@njit
def etaL(en: np.ndarray) -> np.ndarray:
    denom  = (en+me)*(en+me) - me*me
    return -2.0*alpha*me/np.sqrt(denom)

@njit
def aL(en: np.ndarray) -> np.ndarray:
    eta = etaL(en)
    nom = np.exp(2.0*eta*np.arctan(-2.0/eta))
    denom = (1.0 + eta*eta/4.0)*(1.0 + eta*eta/4.0)
    return eta*eta*eta*eta*nom/denom
    
@njit
def Lev(n: int, en: np.ndarray) -> np.ndarray:
    al = aL(en)

    if n==2:  # special case
        return 0.25 * (1.0 + al*al - al)
    term1 = 256.0*np.power(n, 5)*np.power(n-2, 2*n-4)/np.power(n+2, 2*n+4)
    term2 = al*al/(n*n*n) - 16.0*n*al*np.power(n-2, n-2)/np.power(n+2, n+2)
    return 2.0*(term1+term2)

# Full (SM+sterile) differential decay rate as function of electron energy [keV].
# Includes all corrections (no continuum orbital e- states) and 
# the first 5 bound discrete atomic states.
def dGammadE(order: bool, munu: float, mN: float, eta: float, en: np.ndarray) -> np.ndarray:
    sum = np.zeros(en.shape)
    for n in range(1,6):
        sum += Lev(n, en) * Diff4nu(order, mN, eta, en, munu, n)
    return sum

# Set up all contributions including continous states and first 5 
# discrete states. Here consider state n as double for continuum states.
@njit
def endCont(munu: float, n: float) -> float:
    return endAt(munu, 1)-4.0*Ryd*(1.0+1.0/(n*n))

def CorrCont(en: np.ndarray, munu: float, n: float) -> np.ndarray:
    e0 = endCont(munu, n)
    nom  = (en+me)*(en+me) - me*me
    fac = Fermi(2,np.sqrt(nom)/(en+me))*S(2,en)*G(en,e0)
    return fac*L(2,en)*CC(2,en,e0)*Q(2,en,e0)

def DiffCont(en: np.ndarray, munu: float, n: float) -> np.ndarray:
    e0 = endCont(munu, n)
    fac = stub(en, munu, e0)
    return fac * CorrCont(en, munu, n)    

# integrand over states g
def integrand(g: float, en: np.ndarray, munu: float) -> np.ndarray:
    fac1 = DiffCont(en, munu, g) * twopi / (g*g*g * (np.exp(twopi*g) - 1.0))
    fac2 = g*g*g*g * np.exp(2.0 * g * np.arctan(-2.0/g)) / ((1.0 + g*g/4.0)*(1.0 + g*g/4.0))
    return fac1 * (fac2*fac2 + aL(en)*aL(en) - aL(en)*fac2)
    

# Contribution only from the continuum orbital electron states.
# integrate over g.
def gammaCont(enin: np.ndarray, munu: float) -> np.ndarray:
    msk = np.logical_and(enin<endCont(munu, 1e10), enin>=emin)
    res = np.zeros(enin.shape)  # =0 for (not msk)
    en = np.copy(enin[msk])  # shorter than en array
    if en.size<1:
        return res
    # Can't be vectorized due to integral upper limit etaL(en)
    coll = []
    for e in en:  # work-around array input
        coll.append(quad(integrand, -99, etaL(np.array([e])), args=(np.array([e]), munu))[0]/pi)
    res[msk] = np.array(coll)  # same shape
    return res

def dGammadECont(order: bool, munu: float, mN: float, eta: float, en: np.ndarray) -> np.ndarray:
    if order:
        UeSq = UeSqNO
    else:
        UeSq = UeSqIO
    sum = np.zeros(en.shape)
    for i in range(3):
        sum += UeSq[i] * gammaCont(en, nuSpectrum(order,munu)[i])
    term2 = eta*eta*gammaCont(en, mN)
    return (1-eta*eta)*sum + term2


def dGammadEFull(order: bool, munu: float, mN: float, eta: float, en: np.ndarray) -> np.ndarray:
    return dGammadE(order, munu, mN, eta, en) + dGammadECont(order, munu, mN, eta, en)


#
# Set up single parameter, munu, PDF
#
def KurieTrsf(en: np.ndarray, y: np.ndarray) -> np.ndarray:
    ''' Kurie transformation with existing arrays. '''
    momentum = np.sqrt(en * (en + 2.0*me))
    arg = momentum/(en+me)
    Fermif = Fermi(2, arg)  # beta from p/m
    return np.sqrt(y / ((en + me)*momentum*Fermif))


def KuriePDF(en: np.ndarray, munu: float, config: tuple) -> np.ndarray:
    ''' Normalized Kurie spectrum. '''
    order = config[0]
    mN    = config[1]
    eta   = config[2]
    yarr = dGammadE(order, munu, mN, eta, en)
    karr = KurieTrsf(en, yarr)
    norm = np.trapz(karr, en)
    karr /= norm
    return karr
