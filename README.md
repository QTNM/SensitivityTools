# SensitivityTools
Tool collection for sensitivity analyses.

This package is structured in two parts on purpose, the Python folder
contains tools for that language, similarly for the Cpp (C++) folder.

The main tool(s) for now are the Tritium beta decay spectrum calculations
aiming at creating the PDF of the decay model. Other tools may then use these
PDF calculations for sensitivity scripts/codes. These could involve Python
packages or ROOT for statistical analyses.

Note: the model spectrum is invalid for non-physical negative neutrino masses
(the maths is invalid for that case), and the implementation eliminates 
the spectrum endpoint as free variable, on purpose.

## Cpp part: Build instruction

At Warwick, SCRTP, use cvmfs as the easiest environment setup (with bash):

```
source /cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el9-gcc14-opt/setup.sh
```

which sets up ROOT 6.34 on a CentOS9 foundation. ROOT interfaces for RooFit change after 6.30 and 
have been adapted to newer versions here hence the 6.34 setup. Just create a 'build' directory, then 

```
cd build; cmake ..; make
```
