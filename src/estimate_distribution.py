from model import ESN
from datetime import datetime
import statsmodels.api as sm
import numpy as np

def estimate_distribution(data,index,hyperpara):
    index.test_start = data.nTime - 365*24*2 # 2015-1-1 00:00
    index.test_end = data.nTime - 365*24 - 1 # 2015-12-31 23:00 (inclusive)
    
    esn_model = ESN(data,index)

    esn_model.train(hyperpara.parameter)
    print('ESN model trained with parameters: ', hyperpara.parameter)
    
    t0 = datetime.now()
    esn_model.forecast()
    t1 = datetime.now()

    print('Elapased time: ', t1-t0)
    
    esn_model.compute_forecast_error()
    
    quantile = np.arange(0.025,1,0.025)
    
    forErrorQuantile = np.ndarray((quantile.size, *esn_model.forError.shape[1:]))
    
    for i in range(quantile.size):
        forErrorQuantile[i] = np.nanquantile(esn_model.forError, quantile[i], axis = 0)
    
    return forErrorQuantile, esn_model.forMean