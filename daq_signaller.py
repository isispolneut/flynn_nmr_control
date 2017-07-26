import sys
import numpy as np
import numpy

from time import sleep

from daq_ui import Ui_MainWindow

# Find relevant documentation from Instrumental-lib at
# http://instrumental-lib.readthedocs.io/en/latest/ni-daqs.html
#from instrumental.drivers.daq.ni import NIDAQ, Task, AnalogIn
#from instrumental import u
from PyDAQmx import *

from PyQt5 import QtWidgets, QtCore

import matplotlib.pyplot as plt

from scipy.signal import butter, lfilter, freqz, decimate
from scipy.optimize import curve_fit

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """


    if window_len<3:
        return x

    s=numpy.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=numpy.convolve(w/w.sum(),s,mode='valid')
    return y

def butter_lowpass(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = np.divide(cutoff, nyq)
    b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)
    y = lfilter(b,a,data)
    return y

def exp_dec(t, V, t_0, w, p):
    return 0.5 * V * np.exp(-(t/t_0)) * np.sin(w*t + p)

class PulseGenerator():
    def __init__(self):
        self.waveforms = {'Sinusoidal': self.gen_sin, 'Sawtooth': self.gen_saw, 'Hat':self.gen_hat}

    def generate_pulse(self, waveform, pulse_density):
        return (np.sin(2*np.pi*np.linspace(0,1,pulse_density,endpoint=False)).tolist())

    def gen_sin(self,pulse_density):
        return None

    def gen_saw(self,pulse_density):
        return np.linspace(0,1,pulse_density).tolist()

    def gen_hat(self,pulse_density):
        zero = np.zeros((pulse_density,))
        for i in range(int(pulse_density/4),int(pulse_density*3/4)):
            zero[i]+=1
        return zero

class ApplicationWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        super(ApplicationWindow, self).setupUi(self)

        self.data = []

        self.pulse_button.clicked.connect(self.send_pulse)
        self.export_fid_button.clicked.connect(self.export_fid)

        self.MAX_RATE = 1000000

    def export_fid(self):
        python_fid = [np.multiply(np.arange(0,len(self.data)),1e-6),self.data]
        lv_fid = list(zip(*(np.loadtxt('fid_lv',delimiter='\t'))))

        popt, pcov = curve_fit(exp_dec, python_fid[0], python_fid[1])
        plt.plot(python_fid[0],exp_dec(python_fid[1],*popt),'r.')
        plt.plot(python_fid[0],python_fid[1],'b.')
        plt.show()

    def send_pulse(self):
        # Construct pulse and ensure it meets requirements
        pulse_frequency = self.pulse_frequency_spin.value()
        pulse_duration  = self.pulse_duration_spin.value()
        pulse_density   = self.pulse_density_spin.value()
        pulse_amplitude = self.pulse_amplitude_spin.value()


        return_pulse_duration = self.return_pulse_duration_spin.value()

        # PulseGenerator helper class can be used to create arbitrary switchable
        # waveforms. Currently hardcoded to produce a sinusoidal waveform.

        pulse_generator = PulseGenerator()

        pulse = np.array((pulse_generator.generate_pulse('Sinusoidal',pulse_density)),dtype=np.float64)

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
        # Plot result from device

        # Standard numpy DFT,
        # see https://docs.scipy.org/doc/numpy/reference/routines.fft.html

        # As per Parnell 2008 the signal is multiplied by a sine wave near the Larmor freq
        # to create a low frequency beat pattern which is then filtered via low pass
        # Butterworth filter to obtain a clean signal for the DFT.

        #samp_rate = self.MAX_RATE

        larmor_wave = np.cos(2*np.pi*1e-3*np.arange(0,len(self.data))*self.pulse_frequency_spin.value())
        data = np.multiply(self.data,larmor_wave)
        data_filtered = butter_lowpass(data,300,60e3,order=5)

        self.data = data_filtered
        self.plot_return_pulse()

    def plot_return_pulse(self):
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
    aw = ApplicationWindow()

    aw.show()
    sys.exit(qApp.exec_())
