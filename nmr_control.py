import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
from scipy.signal import butter, lfilter, freqz, decimate
from scipy.optimize import curve_fit

from PyDAQmx import *

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QFileDialog
from daq_ui import Ui_MainWindow

from QPlot import QPlot

def butter_lowpass(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = np.divide(cutoff, nyq)
    b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)
    y = lfilter(b,a,data)
    return y

def exp_dec(t, vpp, decay_time, frequency, phase_diff, constant):
    return 0.5*vpp*np.exp(-(t/decay_time))*np.sin(2*np.pi*frequency*t+phase_diff)+constant

class NMRControl(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        super(NMRControl, self).setupUi(self)
        self.setup_override()

        self.data = []
        self.series_amplitudes = []

        # Single FID Acquisition Signals

        def send_pulse():
            self.send_pulse(self.pulse_frequency_spin.value(),
                            self.pulse_duration_spin.value(),
                            self.pulse_density_spin.value(),
                            self.pulse_amplitude_spin.value(),
                            self.return_pulse_duration_spin.value())

        self.pulse_button.clicked.connect(send_pulse)
        self.export_fid_button.clicked.connect(self.export_fid)
        self.import_fid_button.clicked.connect(self.import_fid)

        # Single FID Fitting Signals

        self.fit_button.clicked.connect(lambda: self.fit_fid(plot=True))

        # FID Series Signals

        self.fid_dir = ""
        self.fid_directory_button.clicked.connect(self.set_fid_dir)
        self.afid_start_button.clicked.connect(self.get_fid_series)
        self.fid_directory_button_2.clicked.connect(self.set_fid_dir)
        self.plot_fid_series_button.clicked.connect(self.fit_fid_series)
        self.export_fid_series_button.clicked.connect(self.export_fid_series_fit)

        self.MAX_RATE = 1000000

    def set_fid_dir(self):
        fn = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.fid_directory_display.setText(fn)
        self.fid_directory_display_2.setText(fn)
        self.fid_dir = fn

    def get_fid_series(self):
        # timer must be declared as a member variable so that it doesn't
        # go out of scope after the succesful execution of get_fid_series
        # same with i
        self.timer = QtCore.QTimer()
        self.timer_timer = QtCore.QTimer()

        self.i=0
        def get_and_save():
            self.send_pulse(self.apulse_frequency_spin.value(),
                            self.apulse_duration_spin.value(),
                            self.apulse_density_spin.value(),
                            self.apulse_amplitude_spin.value(),
                            self.areturn_pulse_duration_spin.value())
            self.export_fid(filename=self.fid_dir + '/' + str(self.i))
            self.i+=1
            if self.i>=self.afid_n_series_spin.value():
                self.timer.stop()

            self.statusbar.showMessage('{} of {} series acquired'.format(self.i,self.afid_n_series_spin.value()))

        def update_progress():
            self.fid_series_acq_progress.setValue((self.fid_series_acq_progress.value()+1)%100)
            
        self.timer.timeout.connect(get_and_save)
        self.timer_timer.timeout.connect(update_progress)
        self.timer.start(self.afid_sampling_period_spin.value()*60*1e3)
        self.timer_timer.start(self.afid_sampling_period_spin.value()*600)

    def fit_fid_series(self):
        self.series_amplitudes = []
        self.i = 0
        self.series_timescale = []
        while self.i < self.afid_n_series_spin_2.value():

            if self.import_fid(filename=self.fid_dir + '/' + str(self.i)):
                self.series_amplitudes.append(self.fit_fid(plot=False)[0])
                self.series_timescale.append(self.afid_sampling_period_spin_2.value()*self.i)
            self.i+=1

        self.fid_series_fitting_plot.plot_figure(self.series_timescale,self.series_amplitudes,format='r.')

    def export_fid_series_fit(self):
        if len(self.series_amplitudes) == 0:
            self.statusbar.showMessage('No FID data collected')
            return

        options = QFileDialog.Options()
        filename, _ = QFileDialog.getSaveFileName(self,
                                                    "QFileDialog.getSaveFileName()",
                                                    "",
                                                    "All Files (*)",
                                                    options=options)
        
        if filename:
            to_write = np.column_stack((self.series_timescale,self.series_amplitudes))
            np.savetxt(filename, to_write)
    
    def setup_override(self):
        # Override QtDesigner compiled settings to create QPlots
        # This function literally exists because I'm too lazy to set up
        # QPlot as a QtDesigner plugin

        self.pulseOutput = QPlot(self.nmr_signaller_tab)
        self.pulseOutput.setGeometry(QtCore.QRect(10, 250, 531, 221))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pulseOutput.sizePolicy().hasHeightForWidth())
        self.pulseOutput.setSizePolicy(sizePolicy)
        self.pulseOutput.setObjectName("pulseOutput")

        self.fidOutput = QPlot(self.nmr_signaller_tab,xlabel="Time / ms",ylabel="Amplitude / V")
        self.fidOutput.setGeometry(QtCore.QRect(10, 20, 531, 221))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.fidOutput.sizePolicy().hasHeightForWidth())
        self.fidOutput.setSizePolicy(sizePolicy)
        self.fidOutput.setObjectName("fidOutput")

        self.fid_fitting_plot = QPlot(self.fitting_tab,xlabel="Time / s",ylabel="Amplitude / V")
        self.fid_fitting_plot.setGeometry(QtCore.QRect(10, 10, 741, 311))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.fid_fitting_plot.sizePolicy().hasHeightForWidth())
        self.fid_fitting_plot.setSizePolicy(sizePolicy)
        self.fid_fitting_plot.setObjectName("fid_fitting_plot")

        self.fid_series_fitting_plot = QPlot(self.series_fitting_tab,xlabel="Time / min",ylabel=r"V_pp")
        self.fid_series_fitting_plot.setGeometry(QtCore.QRect(210, 10, 541, 431))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.fid_series_fitting_plot.sizePolicy().hasHeightForWidth())
        self.fid_series_fitting_plot.setSizePolicy(sizePolicy)
        self.fid_series_fitting_plot.setObjectName("fid_series_fitting_plot")

    def fit_fid(self, plot=True):
        if len(self.data) == 0:
            self.statusbar.showMessage('No FID data collected')
            return

        if self.bounds_checkbox.isChecked():
            lower_bounds = [self.vpp_bound_lower.value(),
                            self.decay_time_bound_lower.value(),
                            self.frequency_bound_lower.value(),
                            0,
                            self.constant_bound_lower.value()]

            upper_bounds = [self.vpp_bound_upper.value(),
                            self.decay_time_bound_upper.value(),
                            self.frequency_bound_upper.value(),
                            6.28,
                            self.constant_bound_upper.value()]

            bounds = (lower_bounds,upper_bounds)
        else:
            bounds = (-np.inf,np.inf)

        if self.initial_checkbox.isChecked():
            p0 = [self.vpp_bound_initial.value(),
                  self.decay_time_bound_initial.value(),
                  self.frequency_bound_initial.value(),
                  0,
                  self.constant_bound_initial.value()]

            # Supply initial phase based on knowledge of signal

            if np.sign(np.gradient(self.data[:1000])[0]) == -1:
                p0[3] = np.pi

        # Standard SciPy curve fitting, see documentation at
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html

        timescale = 1e-6*np.arange(0,len(self.data))
        try:
            popt, pcov = curve_fit(exp_dec,timescale,
                                    self.data,p0=p0,bounds=bounds)
        except ValueError:
            self.statusbar.showMessage('Inappropriate initial or boundary values')
            return

        if plot == True:
            self.fid_fitting_plot.axes.cla()
            self.fid_fitting_plot.axes.plot(timescale, exp_dec(timescale,*popt),'b-')
            self.fid_fitting_plot.axes.plot(timescale, self.data,'r-')
            self.fid_fitting_plot.draw()

            self.vpp_found.setText(str(popt[0]*1e3) + ' mV')
            self.decay_constant_found.setText(str(popt[1]*1e3) + ' ms')
            self.frequency_found.setText(str(popt[2]) + ' Hz')
            self.constant_found.setText(str(popt[4]))

        return popt

    def export_fid(self, filename=None):
        if not filename:
            if len(self.data) == 0:
                self.statusbar.showMessage('No FID data collected')
                return

            options = QFileDialog.Options()
            filename, _ = QFileDialog.getSaveFileName(self,
                                                        "QFileDialog.getSaveFileName()",
                                                        "",
                                                        "All Files (*)",
                                                        options=options)
 
        if filename:
            timescale = np.multiply(np.arange(0,len(self.data)),1e-6)
            to_write = np.column_stack((timescale,self.data))
            np.savetxt(filename, to_write)

    def import_fid(self, filename=None):
        if not filename:
            options = QFileDialog.Options()
            filename, _ = QFileDialog.getOpenFileName(self,
                                                    "QFileDialog.getOpenFileName()",
                                                    "",
                                                    "All Files (*)",
                                                    options=options)

        if filename:
            try:
                self.data = list(zip(*np.loadtxt(filename)))[1]
                self.plot_return_pulse()
                return True
            except FileNotFoundError:
                return False

    def send_pulse(self, pulse_frequency, pulse_duration, pulse_density, pulse_amplitude, return_pulse_duration):
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

        self.fid_fitting_plot.axes.cla()
        self.fid_fitting_plot.axes.plot(1e-6*np.arange(0,len(self.data)), self.data,'r-')

        # Formatting arguments for plots, perhaps in future this can be implemented in an options dialog?

        self.pulseOutput.axes.set_xlim([0,2000])
        #self.pulseOutput.axes.set_ylim([0,2.5])
        #self.fidOutput.axes.set_ylim([-0.2,0.2])
        self.pulseOutput.draw()
        self.fidOutput.draw()
        self.fid_fitting_plot.draw()
        
if __name__ == '__main__':
    qApp = QtWidgets.QApplication(sys.argv)
    aw = NMRControl()

    aw.show()
    sys.exit(qApp.exec_())
