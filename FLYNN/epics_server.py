from pcaspy import Driver
from random import random
from numpy import around, asarray

class flynnDriver(Driver):
    def __init__(self, nmr_controller=None):
        super(flynnDriver, self).__init__()
        if nmr_controller:
            self.nmr_controller = nmr_controller

    def read(self, reason):

        if reason == 'STATE':
            return self.getParam('STATE')

        elif reason == 'FID':
            # self.nmr_controller.wrapd_send_pulse()
            test = asarray([1.234, 1.456, 1.789, 3.555])
            self.setParamInfo('FID',{'count': len(test)})
            self.setParam('FID',test)
            self.updatePVs()
            return test

    def write(self, reason, value):

        if reason == 'STATE':
            
            if type(value) is int:
                state = self.getParam('STATE')
                if state != value:
                    print('frequency = '+str(self.nmr_controller.larmor_freq_spin.value()))
                    print('range = '+str(self.nmr_controller.ramp_range_spin.value()))
                    print('period = '+str(self.nmr_controller.afp_period_spin.value()))
                    print('amplitude = '+str(self.nmr_controller.afp_amplitude_spin.value()))
                    # self.nmr_controller.afp_flip_wrap()
                    self.setParam('STATE',value)
                    self.updatePVs()
                    print('state changed to '+str(value))
                else:
                    print('3He already in correct state')

                return state
            else:
                print('Bad value')
                return state