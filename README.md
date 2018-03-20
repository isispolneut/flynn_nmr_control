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

## Compiling executable

Open the folder FLYNN and run

```
python setup.py build
```

This will require you to have the above dependencies installed. You might also need to wrangle with the many, many issues with cx_freeze - these issues can be resolved via liberal application of Google-fu. A binary is provided in this repository compiled on win64.

## Compiling .ui files to .py

Run the following in a python interpreter in the directory of the .ui files:

```
import PyQt5.uic
PyQt5.uic.compileUiDir('.')
```

After altering the .ui file make sure to update the setup_override function in nmr_control.py.