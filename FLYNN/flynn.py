import numpy as np
from scipy.signal import butter, lfilter, freqz, decimate
from scipy.optimize import curve_fit
from PyDAQmx import *
from time import time

MAX_RATE = 1000000

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    from math import factorial
    
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

def exp_dec(t, vpp, decay_time, b, frequency, constant, phase_diff):
    return 0.5*vpp*np.exp(-(t/decay_time)**(1+b**2))*np.sin(2*np.pi*frequency*t+phase_diff)+constant

def butter_lowpass(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = np.divide(cutoff, nyq)
    b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)
    y = lfilter(b,a,data)
    return y

def afp_pulse_form(t,wl,wh,wlarmor,l,amplitude):
    # Outputs a Gaussian modulated sine wave whose
    # frequency is linearly interpolated over the
    # pulse period between wl and wh.
    w = t*(np.abs(wh-wl)/(l)) + wl
    mu = wlarmor
    sigma = wlarmor*0.14
    gauss = np.exp(-((w - mu)**2)/(sigma**2))
    return amplitude*gauss*np.sin(2*np.pi*w*t)

def afp_flip(wl,wh,w_larmor,l,amplitude,out='Dev2/ao1',readback='Dev2/ai6'):
        t = np.linspace(0,l,num=l*MAX_RATE)
        w = t*(np.abs(wh-wl)/(l)) + wl
        pulse = afp_pulse_form(t,wl*1e3,wh*1e3,w_larmor*1e3,l,amplitude)

        analog_output = Task()
        wrote = int32()

        analog_output.CreateAOVoltageChan(out,
                                           "",
                                           -2,2,
                                           DAQmx_Val_Volts,
                                           None)

        analog_output.CfgSampClkTiming("",
                                        MAX_RATE,
                                        DAQmx_Val_Rising,
                                        DAQmx_Val_FiniteSamps,
                                        len(pulse))

        analog_output.WriteAnalogF64(len(pulse),
                                      False,
                                      10.0,
                                      DAQmx_Val_GroupByChannel,
                                      pulse,
                                      wrote,
                                      None)
        
        analog_input = Task()
        data = np.zeros((int(MAX_RATE*l),),dtype=np.float64)
        read = int32()

        analog_input.CreateAIVoltageChan(readback,
                                          "",
                                          -1,
                                          -2,2,
                                          DAQmx_Val_Volts,
                                          None)

        analog_input.CfgSampClkTiming("",
                                       MAX_RATE,
                                       DAQmx_Val_Rising,
                                       DAQmx_Val_FiniteSamps,
                                    int(MAX_RATE*l))


        
        analog_input.StartTask()
        analog_output.StartTask()
        

        analog_input.ReadAnalogF64(-1,
                                    10.0,
                                    DAQmx_Val_GroupByChannel,
                                    data,
                                    len(data),
                                    read,
                                    None)

        analog_output.ClearTask()
        analog_input.ClearTask()

def send_pulse(pulse_frequency, pulse_duration, 
                pulse_density, pulse_amplitude, return_pulse_duration,
                input_terminal='Dev2/ao0', output_terminal='Dev2/ai0'):
    # Ensure sampling rate is below the maximum rate for the current 1DAQ, 1MHz

    if pulse_frequency*pulse_density > 1e3:
        print('Requested sampling rate too high')
        return None

    # DAQmx API calls to pulse for pulse_duration and then read for return_pulse_duration
    # C API documentation can be found in National Instruments folder. Python API wraps
    # C API calls 1 to 1.

    analog_output = Task()
    analog_input  = Task()
    wrote = int32()
    read  = int32()
    data = np.zeros((int(return_pulse_duration*1e3),),dtype=np.float64)
    pulse = pulse_amplitude*np.sin(2*np.pi*np.linspace(0,1,pulse_density,endpoint=False))

    analog_output.CreateAOVoltageChan(input_terminal,
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
    analog_input.CreateAIVoltageChan(output_terminal,
                                      "",
                                      -1,
                                      -0.2,0.2,
                                      DAQmx_Val_Volts,
                                      None)
    analog_input.CfgSampClkTiming("",
                                   MAX_RATE,
                                   DAQmx_Val_Rising,
                                   DAQmx_Val_FiniteSamps,
                                   int(return_pulse_duration*1e3))
    analog_input.CfgAnlgEdgeStartTrig(output_terminal,
                                    DAQmx_Val_RisingSlope,
                                    0.01)
    analog_input.StartTask()
    analog_output.StartTask()

    analog_input.ReadAnalogF64(-1,
                                10.0,
                                DAQmx_Val_GroupByChannel,
                                data,
                                len(data),
                                read,
                                None)
   
    analog_output.ClearTask()
    analog_input.ClearTask()

    return data[2150:]