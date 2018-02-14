from flynn import *

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
from scipy.optimize import curve_fit
from scipy.signal import resample
from scipy.stats import linregress

from PyDAQmx import *

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QFileDialog
from daq_ui import Ui_MainWindow
from params_dialog import ParamsDialog

from QPlot import QPlot

from epics_server import flynnDriver
from pcaspy import SimpleServer
from pcaspy.tools import ServerThread

class NMRControl(QtWidgets.QMainWindow, Ui_MainWindow):
    """Main Qt window class"""

    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        super(NMRControl, self).setupUi(self)

        self.setup_override()

        self.acq_data = []  # Holds the most recent FID amplitude data
        self.timescale = [] # Holds the most recent FID time data
        self.fit_data = []  # Holds the most recent IMPORTED FID ...
        self.fit_time = []  # ...
        self.fit_params = []# Holds the time series of fitted FID parameters
        self.fit_error = [] # Holds the time series of fitted FID errors

        self.i = 0

        # Single FID Acquisition Signals

        def wrapd_send_pulse():
            # There are neater syntaxes for this than an inline function
            # wrapper but this works for now
            self.acq_data = send_pulse(self.pulse_frequency_spin.value(),
                                       self.pulse_duration_spin.value(),
                                       self.pulse_density_spin.value(),
                                       self.pulse_amplitude_spin.value(),
                                       self.return_pulse_duration_spin.value(),
                                       input_terminal=self.input_terminal_combo.currentText(),
                                       output_terminal=self.output_terminal_combo.currentText())

            self.filter_return_pulse()

        self.pulse_button.clicked.connect(wrapd_send_pulse)
        self.export_fid_button.clicked.connect(self.export_fid)
        self.import_fid_button.clicked.connect(self.import_fid)

        # Single FID Fitting Signals

        self.fit_button.clicked.connect(lambda: self.fit_fid(plot=True))

        # FID Series Signals

        self.fid_dir = ""
        self.fid_directory_button.clicked.connect(self.set_fid_dir)
        self.afid_start_button.clicked.connect(self.get_fid_series)
        self.fid_directory_button_2.clicked.connect(self.set_fid_dir)
        self.fit_fid_series_button.clicked.connect(self.fit_fid_series)
        self.export_fid_series_button.clicked.connect(self.export_fid_series_fit)
        self.plot_multiple_fid_button.clicked.connect(self.plot_multiple_fid)

        def params_dialog():
            # Brings up the parameters dialog
            pd = ParamsDialog(self)
            pd.show()

        self.fid_series_params_button.clicked.connect(params_dialog)

        self.MAX_RATE = 1000000
        self.state = 1

    def set_fid_dir(self):
        """Sets the directory from which to read and write FID series"""

        fn = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.fid_directory_display.setText(fn)
        self.fid_directory_display_2.setText(fn)
        self.fid_dir = fn

    def get_fid_series(self):
        """Starts recording a time series of FIDs"""

        # timer must be declared as a member variable so that it doesn't
        # go out of scope after the succesful execution of get_fid_series
        # same with i
        self.timer = QtCore.QTimer()
        self.timer_timer = QtCore.QTimer()

        self.i = 0

        # Wrapped function to attach to the acquisition timer timeout
        def get_and_save():
            self.acq_data = send_pulse(self.apulse_frequency_spin.value(),
                                       self.apulse_duration_spin.value(),
                                       self.apulse_density_spin.value(),
                                       self.apulse_amplitude_spin.value(),
                                       self.areturn_pulse_duration_spin.value())
            self.filter_return_pulse()
            self.export_fid(filename=self.fid_dir + '/' + str(self.i))
            self.i += 1
            if self.i >= self.afid_n_series_spin.value():
                self.timer.stop()
                self.timer_timer.stop()

            self.statusbar.showMessage(
                '{} of {} series acquired'.format(
                    self.i, self.afid_n_series_spin.value()))

        # Wrapped function to run the progress bar. Because of processing
        # times this falls out of sync with the actual progress but I'm going
        # to keep it around just because it shows that the program is actually
        # working.
        def update_progress():
            self.fid_series_acq_progress.setValue(
                (self.fid_series_acq_progress.value() + 1) % 100)

        self.timer.timeout.connect(get_and_save)
        self.timer_timer.timeout.connect(update_progress)
        self.timer.start(self.afid_sampling_period_spin.value() * 60 * 1e3)
        self.timer_timer.start(self.afid_sampling_period_spin.value() * 600)

    def fit_fid_series(self):
        """Fits N series taken from directory specified by set_fid_dir()"""

        series_amplitudes = []
        series_timescale = []
        self.fit_error = []
        self.fit_params = []
        self.i = 0

        while self.i < self.afid_n_series_spin_2.value():

            if self.import_fid(filename=self.fid_dir + '/' +
                               str(self.afid_file_prefix.text()) + str(self.i)):
                try:
                    fit_result = self.fit_fid(plot=False)

                    # Save results for plotting
                    series_amplitudes.append(fit_result[0][0])
                    series_timescale.append(
                        self.afid_sampling_period_spin_2.value() * self.i)

                    # Append time to fitting parameters
                    fit_result[0].append(
                        self.afid_sampling_period_spin_2.value() * self.i)

                    # Record fitting parameters in time series
                    self.fit_error.append(np.diag(fit_result[1]))
                    self.fit_params.append(fit_result[0])
                    print("Fit FID # " + str(self.i))
                except RuntimeError:
                    # If a fit isn't found for a given FID with the supplied parameters, simply skip the file
                    # This will leave a gap but it at least won't crash the whole routine
                    # for now.
                    print("Unable to fit FID #" + str(self.i))
                    self.i += 1
                    continue
            self.i += 1

        self.fid_series_fitting_plot.axes.cla()
        self.fid_series_fitting_plot.axes.set_xlabel("time/min")
        self.fid_series_fitting_plot.axes.set_ylabel("amplitude/V")
        self.fid_series_fitting_plot.axes.plot(series_timescale, series_amplitudes, 'r.')
        self.fid_series_fitting_plot.draw()

    def export_fid_series_fit(self):
        """Exports the parameters of a fitted FID time series"""

        if len(self.fit_params) == 0:
            self.statusbar.showMessage('No FID data collected')
            return

        options = QFileDialog.Options()
        filename, _ = QFileDialog.getSaveFileName(self,
                                                  "QFileDialog.getSaveFileName()",
                                                  "",
                                                  "All Files (*)",
                                                  options=options)

        if filename:
            np.savetxt(filename, self.fit_params)
            np.savetxt(filename + "_err", self.fit_error)

    def setup_override(self):
        """Override QtDesigner compiled settings to create QPlots. This function exists
        because I'm too lazy to set up QPlot as a QtDesigner plugin
        """

        self.pulseOutput = QPlot(self.nmr_signaller_tab)
        self.pulseOutput.setGeometry(QtCore.QRect(10, 250, 531, 221))
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pulseOutput.sizePolicy().hasHeightForWidth())
        self.pulseOutput.setSizePolicy(sizePolicy)
        self.pulseOutput.setObjectName("pulseOutput")

        self.fidOutput = QPlot(
            self.nmr_signaller_tab,
            xlabel="Time / ms",
            ylabel="Amplitude / mV")
        self.fidOutput.setGeometry(QtCore.QRect(10, 20, 531, 221))
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.fidOutput.sizePolicy().hasHeightForWidth())
        self.fidOutput.setSizePolicy(sizePolicy)
        self.fidOutput.setObjectName("fidOutput")

        self.fid_fitting_plot = QPlot(
            self.fitting_tab,
            xlabel="Time / s",
            ylabel="Amplitude / V")
        self.fid_fitting_plot.setGeometry(QtCore.QRect(10, 10, 741, 261))
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.fid_fitting_plot.sizePolicy().hasHeightForWidth())
        self.fid_fitting_plot.setSizePolicy(sizePolicy)
        self.fid_fitting_plot.setObjectName("fid_fitting_plot")

        self.fid_series_fitting_plot = QPlot(
            self.series_fitting_group,
            xlabel="Time / min",
            ylabel=r"V_pp")
        self.fid_series_fitting_plot.setGeometry(QtCore.QRect(190, 20, 541, 191))
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.fid_series_fitting_plot.sizePolicy().hasHeightForWidth())
        self.fid_series_fitting_plot.setSizePolicy(sizePolicy)
        self.fid_series_fitting_plot.setObjectName("fid_series_fitting_plot")

        self.fid_multiple_plot = QPlot(
            self.multiple_plotting_group,
            xlabel="Time / ms",
            ylabel="Amplitude / mV")
        self.fid_multiple_plot.setGeometry(QtCore.QRect(190, 20, 541, 211))
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.fid_multiple_plot.sizePolicy().hasHeightForWidth())
        self.fid_multiple_plot.setSizePolicy(sizePolicy)
        self.fid_multiple_plot.setObjectName("fid_multiple_plot")

    def fit_fid(self, plot=True):
        """Fits an FID using scipy.optimize.curve_fit()"""

        if len(self.fit_data) == 0:
            self.statusbar.showMessage('No FID data collected')
            return

        # Pull fitting parameters from the UI
        is_fixed_mask = [bool(self.vpp_bound_fixed_check.checkState()),
                         bool(self.decay_time_bound_fixed_check.checkState()),
                         False,
                         bool(self.frequency_bound_fixed_check.checkState()),
                         bool(self.constant_bound_fixed_check.checkState()),
                         False]

        fixed_bounds = [self.vpp_bound_fixed.value() * 1e-3,
                        self.decay_time_bound_fixed.value() * 1e-3,
                        0,
                        self.frequency_bound_fixed.value(),
                        self.constant_bound_fixed.value(),
                        0]

        if self.bounds_checkbox.isChecked():
            lower_bounds = np.array([self.vpp_bound_lower.value() * 1e-3,
                                     self.decay_time_bound_lower.value() * 1e-3,
                                     0,
                                     self.frequency_bound_lower.value(),
                                     self.constant_bound_lower.value(),
                                     0])
            lower_bounds = lower_bounds[~np.array(is_fixed_mask)]
            upper_bounds = np.array([self.vpp_bound_upper.value() * 1e-3,
                                     self.decay_time_bound_upper.value() * 1e-3,
                                     1,
                                     self.frequency_bound_upper.value(),
                                     self.constant_bound_upper.value(),
                                     6.28])
            upper_bounds = upper_bounds[~np.array(is_fixed_mask)]

            bounds = (lower_bounds, upper_bounds)
        else:
            bounds = (-np.inf, np.inf)

        if self.initial_checkbox.isChecked():
            p0 = np.array([self.vpp_bound_initial.value() * 1e-3,
                           self.decay_time_bound_initial.value() * 1e-3,
                           0,
                           self.frequency_bound_initial.value(),
                           self.constant_bound_initial.value(),
                           0])
            p0 = p0[~np.array(is_fixed_mask)]
        else:
            p0 = None

        # This is one of the hackiest dumbest sections of code I've ever written.
        # To future me or anyone else that ever has to improve or maintain this,
        # I apologise.

        # Basically to save having to manually create lambdas for all 16 possible
        # permutations of the arguments for exp_dec, I have built a system whereby
        # I construct the lambda dynamically as a string and then create it using
        # eval().

        popt_amended = [None, None, None, None, None, None]
        params_st = ['vpp', 'decay_time', 'b', 'frequency', 'constant', 'phase_diff']
        params = ['vpp', 'decay_time', 'b', 'frequency', 'constant', 'phase_diff']
        i = len(params) - 1
        for m in is_fixed_mask[::-1]:
            if m:
                del params[i]
            i -= 1

        fitting_func_str = 'lambda t,' + ','.join(params) + ': exp_dec(t,'
        i = 0
        for m in is_fixed_mask:
            if m:
                fitting_func_str += str(fixed_bounds[i]) + ','
                popt_amended[i] = fixed_bounds[i]
            else:
                fitting_func_str += params_st[i] + ','
            i += 1
        fitting_func_str = fitting_func_str[:-1]  # Remove trailing comma
        fitting_func_str += ')'

        fitting_func = eval(fitting_func_str)

        # Standard SciPy curve fitting, see documentation at
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html

        try:
            popt, pcov = curve_fit(fitting_func, self.fit_time,
                                   self.fit_data, p0=p0, bounds=bounds, max_nfev=600)
        except ValueError:
            self.statusbar.showMessage('Inappropriate initial or boundary values')
            return
        except RuntimeError:
            self.statusbar.showMessage('Unable to determine FID parameters')
            return

        # Merge the fixed and determined parameter lists
        i = 0
        for p in popt_amended:
            if p is None:
                popt_amended[popt_amended.index(p)] = popt[i]
                i += 1
        popt = popt_amended

        # Plot the fit if requested and update the found parameters in the UI
        if plot == True:
            self.fit_time = np.array(self.fit_time)
            self.fid_fitting_plot.axes.cla()
            self.fid_fitting_plot.axes.plot(
                self.fit_time, exp_dec(
                    self.fit_time, *popt), 'b-')
            self.fid_fitting_plot.axes.plot(self.fit_time, self.fit_data, 'r-')
            self.fid_fitting_plot.draw()

            self.vpp_found.setText(str(np.round(popt[0] * 1e3, decimals=3)) + ' mV')
            self.decay_constant_found.setText(
                str(np.round(popt[1] * 1e3, decimals=2)) + ' ms')
            self.frequency_found.setText(str(np.round(popt[3], decimals=0)) + ' Hz')
            self.constant_found.setText(str(popt[4]))

        # Return results of the fit
        return [popt, pcov]

    def export_fid(self, filename=None):
        """Exports the time/amplitude of an FID"""

        if not filename:
            if len(self.acq_data) == 0:
                self.statusbar.showMessage('No FID data collected')
                return

            options = QFileDialog.Options()
            filename, _ = QFileDialog.getSaveFileName(self,
                                                      "QFileDialog.getSaveFileName()",
                                                      "",
                                                      "All Files (*)",
                                                      options=options)

        if filename:
            to_write = np.column_stack((self.timescale, self.acq_data))
            np.savetxt(filename, to_write)

    def import_fid(self, filename=None):
        """Imports the time/amplitude of an FID"""

        if not filename:
            options = QFileDialog.Options()
            filename, _ = QFileDialog.getOpenFileName(self,
                                                      "QFileDialog.getOpenFileName()",
                                                      "",
                                                      "All Files (*)",
                                                      options=options)

        if filename:
            try:
                data = list(zip(*np.loadtxt(filename)))
                self.fit_data = data[1]
                self.fit_time = data[0]
                self.fid_fitting_plot.axes.cla()
                self.fid_fitting_plot.axes.plot(
                    1e-6 * np.arange(0, len(self.fit_data)), self.fit_data, 'r-')
                self.fid_fitting_plot.draw()
                return True
            except FileNotFoundError:
                return False

    def plot_multiple_fid(self):
        try:
            text = self.multiple_fids_input.text()
            vals = text.split(sep=',')
            fids = []
            for v in vals:
                if v.find('-') == -1:
                    fids.append(v)
                else:
                    bounds = v.split('-')
                    for i in list(range(int(bounds[0]), int(bounds[1]) + 1)):
                        fids.append(str(i))
            self.fid_multiple_plot.axes.cla()
            for fid in fids:
                data = list(zip(*np.loadtxt(self.fid_dir + '/' + fid)))
                self.fid_multiple_plot.axes.plot(data[0], np.multiply(data[1], 1e3))
            self.fid_multiple_plot.axes.set_xlabel(self.fid_multiple_plot.xlabel)
            self.fid_multiple_plot.axes.set_ylabel(self.fid_multiple_plot.ylabel)
            self.fid_multiple_plot.draw()
        except (ValueError, FileNotFoundError) as err:
            print(err)
            self.statusbar.showMessage('Invalid FID inputs')
            return
        except PermissionError:
            self.statusbar.showMessage('Invalid directory')
            return

    def filter_return_pulse(self):
        # As per Parnell 2008 the signal is multiplied by a sine wave near the Larmor freq
        # to create a low frequency beat pattern which is then filtered via low pass
        # Butterworth filter to obtain a clean signal for the DFT.

        # Wave frequency in sample time is obtained,
        #
        #    frequency = 2pi*(larmor_frequency/device_sampling_rate)

        larmor_wave = np.cos(2 * np.pi * 1e-3 * np.arange(0,
                                                          len(self.acq_data)) * self.pulse_frequency_spin.value())
        data = np.multiply(self.acq_data, larmor_wave)
        data_filtered = butter_lowpass(data, 300, 60e3, order=5)

        self.acq_data, self.timescale = resample(data_filtered, int(len(
            data_filtered) / self.decimation_factor_spin.value()), t=1e-6 * np.arange(0, len(self.acq_data)))
        self.plot_return_pulse()

    def plot_return_pulse(self):
        # Standard numpy DFT,
        # see https://docs.scipy.org/doc/numpy/reference/routines.fft.html

        read_pulse_fft = np.fft.fft(self.acq_data)
        read_pulse_nu = np.fft.fftfreq(len(self.acq_data),
                                       (self.MAX_RATE) * self.decimation_factor_spin.value())
        read_pulse_fft_shifted = np.fft.fftshift(read_pulse_fft)
        read_pulse_nu_shifted = np.fft.fftshift(read_pulse_nu)

        self.pulseOutput.plot_figure(np.multiply(read_pulse_nu_shifted, 1e12),
                                     np.absolute(read_pulse_fft_shifted)**2)

        self.fidOutput.plot_figure(self.timescale, np.multiply(self.acq_data, 1000))

        # Formatting arguments for plots, perhaps in future this can be
        # implemented in an options dialog?

        self.pulseOutput.axes.set_xlim([0, 2000])
        # self.pulseOutput.axes.set_ylim([0,2.5])
        # self.fidOutput.axes.set_ylim([-0.2,0.2])
        self.pulseOutput.draw()
        self.fidOutput.draw()


if __name__ == '__main__':
    qApp = QtWidgets.QApplication(sys.argv)
    aw = NMRControl()

    prefix = 'NMR:'
    pvdb = {
        'FID': {
            'prec': 7,
        },
        'State': {
            'type': 'int',
            'value': 2,
        }
    }
    server = SimpleServer()
    server.createPV(prefix, pvdb)
    driver = flynnDriver(nmr_controller=aw)

    # create pcas server thread and shut down when app exits
    server_thread = ServerThread(server)
    qApp.lastWindowClosed.connect(server_thread.stop)

    # start pcas and gui event loop
    server_thread.start()

    aw.show()
    sys.exit(qApp.exec_())
