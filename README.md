# flynn_nmr_control
NMR control and FID fitting for the 3He spin filter filling station at the ISIS FLYNN laboratory

## Dependencies

### NMR Control

- Qt
- (Py)DAQmx
- the SciPy stack

### IBEX Interface

- EPICS
- pcaspy

## Minimizing AFP Losses

When flipping 3He polarisation using the adiabatic fast passage signal, use the following parameters to minimize losses:

- First determine the Larmor frequency to a reasonable precision (~5Hz) using the NMR tab
- Set the hi / lo frequencies of the sweep to 7.5KHz either side of the Larmor freq.
- Set the pulse duration to 0.5s
- Keep the pulse amplitude at 0.5V

More optimal parameters are likely possible but this requires further testing. The above settings have been found to give satisfactorily negligible losses.

## Compiling .ui files to .py

Run the following in a python interpreter in the directory of the .ui files:

```import PyQt5.uic
PyQt5.uic.compileUiDir('.')
```