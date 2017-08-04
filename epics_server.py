from pcaspy import Driver
from random import random
from numpy import around, asarray

class flynnDriver(Driver):
    def __init__(self, nmr_controller=None):
        super(flynnDriver, self).__init__()
        if nmr_controller:
            self.nmr_controller = nmr_controller

    def read(self, reason):
        if reason == 'FID':
            self.setParamInfo('FID', 
                            {'count': len(self.nmr_controller.data)})
            self.updatePVs()
            return self.nmr_controller.data
        elif reason == 'State':
            self.nmr_controller.poll_afr_state()
            return self.nmr_controller.state