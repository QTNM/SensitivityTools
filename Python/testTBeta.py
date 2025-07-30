''' test the TBetaGenerator functions. '''
import numpy as np
import matplotlib.pyplot as plt
import TBetaGenerator as tbeta


# script
mnu = 2.0e-5 # keV
n = 1
emax = tbeta.endAt(mnu, n)
print("munu=",mnu," E0=",emax)
emax2 = tbeta.endAt(0.0, n)
print("munu= 0.0"," E0=",emax2)

# spectrum
order = True
mN = 0.0
eta = 0.0
#energy = np.linspace(18.56, emax2, 10000)
energy = np.linspace(0.1, emax2, 10000)

conf = (order, mN, eta)
#karr = tbeta.KuriePDF(energy, mnu, conf)
karr = tbeta.dGammadE(True, 0.0, 0.0, 0.0, energy)

# Not vectorized: very slow
#karr = tbeta.dGammadEFull(True, 0.0, 0.0, 0.0, energy)

plt.plot(energy, karr, 'r-')
plt.show()

