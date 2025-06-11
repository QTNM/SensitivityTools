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
