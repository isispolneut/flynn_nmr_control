import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz, decimate
from scipy.optimize import curve_fit

from PyDAQmx import *

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QFileDialog
from daq_ui import Ui_MainWindow

def butter_lowpass(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = np.divide(cutoff, nyq)
    b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)
    y = lfilter(b,a,data)
    return y

class DAQSignaller(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        super(DAQSignaller, self).setupUi(self)

        self.data = []

        self.pulse_button.clicked.connect(self.send_pulse)
        self.export_fid_button.clicked.connect(self.export_fid)

        self.MAX_RATE = 1000000

    def export_fid(self):
        if len(self.data) == 0:
            self.statusbar.showMessage('No FID data collected')
            return

        timescale = np.multiply(np.arange(0,len(self.data)),1e-6)
        to_write = np.column_stack((timescale,self.data))

        options = QFileDialog.Options()
        filename, _ = QFileDialog.getSaveFileName(self,
                                                    "QFileDialog.getSaveFileName()",
                                                    "",
                                                    "All Files (*)",
                                                    options=options)
        
        np.savetxt(filename, to_write)

    def send_pulse(self):
        # Construct pulse and ensure it meets requirements
        pulse_frequency = self.pulse_frequency_spin.value()
        pulse_duration  = self.pulse_duration_spin.value()
        pulse_density   = self.pulse_density_spin.value()
        pulse_amplitude = self.pulse_amplitude_spin.value()

        return_pulse_duration = self.return_pulse_duration_spin.value()

        # Ensure sampling rate is below the maximum rate for the current DAQ, 1MHz

        self.statusbar.showMessage('') # Clear the status bar
        if pulse_frequency*pulse_density > 1e3:
            self.statusbar.showMessage('Requested sampling rate too high')
            return

        # DAQmx API calls to pulse for pulse_duration and then read for return_pulse_duration
        # C API documentation can be found in National Instruments folder. Python API wraps
        # C API calls 1 to 1.

        analog_output = Task()
        analog_input  = Task()
        wrote = int32()
        read  = int32()
        data = np.zeros((int(return_pulse_duration*1e3),),dtype=np.float64)
        pulse = np.sin(2*np.pi*np.linspace(0,1,pulse_density,endpoint=False))

        analog_output.CreateAOVoltageChan("Dev1/ao0",
                                           "",
                                           -1,1,
                                           DAQmx_Val_Volts,
                                           None)

        analog_output.CfgSampClkTiming("",
                                        pulse_frequency*pulse_density*1e3,
                                        DAQmx_Val_Rising,
                                        DAQmx_Val_FiniteSamps,
                                        int(pulse_frequency*pulse_duration)*pulse_density)

        analog_output.WriteAnalogF64(len(pulse),
                                      False,
                                      10.0,
                                      DAQmx_Val_GroupByChannel,
                                      pulse,
                                      wrote,
                                      None)


        analog_input.CreateAIVoltageChan("Dev1/ai0",
                                          "",
                                          -1,
                                          -0.2,0.2,
                                          DAQmx_Val_Volts,
                                          None)

        analog_input.CfgSampClkTiming("",
                                       self.MAX_RATE,
                                       DAQmx_Val_Rising,
                                       DAQmx_Val_FiniteSamps,
                                       int(return_pulse_duration*1e3))

        analog_output.StartTask()
        analog_input.StartTask()

        analog_input.ReadAnalogF64(-1,
                                    10.0,
                                    DAQmx_Val_GroupByChannel,
                                    data,
                                    len(data),
                                    read,
                                    None)

        analog_output.ClearTask()
        analog_input.ClearTask()

        self.data = data[2000:] # Crop data to remove leading edge from pulse
        self.filter_return_pulse()
        
        
    def filter_return_pulse(self): 
        # As per Parnell 2008 the signal is multiplied by a sine wave near the Larmor freq
        # to create a low frequency beat pattern which is then filtered via low pass
        # Butterworth filter to obtain a clean signal for the DFT.

        # Wave frequency in sample time is obtained,
        #
        #    frequency = 2pi*(larmor_frequency/device_sampling_rate)

        larmor_wave = np.cos(2*np.pi*1e-3*np.arange(0,len(self.data))*self.pulse_frequency_spin.value())
        data = np.multiply(self.data,larmor_wave)
        data_filtered = butter_lowpass(data,300,60e3,order=5)

        self.data = data_filtered
        self.plot_return_pulse()

    def plot_return_pulse(self):
        # Standard numpy DFT,
        # see https://docs.scipy.org/doc/numpy/reference/routines.fft.html

        read_pulse_fft = np.fft.fft(self.data)
        read_pulse_nu  = np.fft.fftfreq(len(self.data),
                                        (self.MAX_RATE))
        read_pulse_fft_shifted = np.fft.fftshift(read_pulse_fft)
        read_pulse_nu_shifted = np.fft.fftshift(read_pulse_nu)

        self.pulseOutput.plot_figure(np.multiply(read_pulse_nu_shifted,1e12),
                                    np.absolute(read_pulse_fft_shifted)**2)

       
        self.fidOutput.plot_figure(0.001*np.arange(0,len(self.data)), np.multiply(self.data,1000))

        # Formatting arguments for plots, perhaps in future this can be implemented in an options dialog?

        self.pulseOutput.axes.set_xlim([0,2000])
        #self.pulseOutput.axes.set_ylim([0,2.5])
        #self.fidOutput.axes.set_ylim([-0.2,0.2])
        self.pulseOutput.draw()
        self.fidOutput.draw()
        
if __name__ == '__main__':
    qApp = QtWidgets.QApplication(sys.argv)
    aw = DAQSignaller()

    aw.show()
    sys.exit(qApp.exec_())
