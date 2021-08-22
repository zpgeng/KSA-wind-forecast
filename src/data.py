import numpy as np

class Data:
    
    def __init__(self, nTime, wind_residual):
        self.nTime = nTime
        self.time = np.arange(nTime)
        self.ts = wind_residual.T