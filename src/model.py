import numpy as np
from scipy.linalg import eigh
from time import perf_counter
import torch

class ESN:
    
    def __init__(self, data, index, device):
        
        self.data = data
        self.index = index
        
        self.ensembleLen = 2 # Ensemble length, default set to 100
        self.numTimePred = 1 # number of time lags to predict, T+1, T+2, ..., T+k
        
        # self.tauEmb = 1 # number of lead time embedded
    
        self.forMeanComputed = False
        self.forErrorComputed = False

        self.device = device
        
    def standardize_in_sample(self, is_validation = False):
        """
        Before regression, we need to standardize the covariates and responses for the Ridge to be valid.
        """

        if(is_validation):
            self.inSampleEmb_len = self.index.validate_start - self.m
        else:
            self.inSampleEmb_len = self.index.test_start - self.m
            
        #### X
        self.inSampleX = np.repeat(np.nan, self.inSampleEmb_len * self.m * self.numLocs).reshape(self.inSampleEmb_len, self.m, -1)
        for i in range(self.inSampleEmb_len):
            self.inSampleX[i, ] = self.data.ts[range(i, (self.m + i), 1)]

        self.inSampleX_mean = self.inSampleX.mean(axis=0)
        self.inSampleX_std = self.inSampleX.std(axis=0)
            
        self.inSampleX = (self.inSampleX - self.inSampleX_mean) / self.inSampleX_std
        self.inSampleDesignMatrix = np.column_stack([np.repeat(1, self.inSampleEmb_len), self.inSampleX.reshape(self.inSampleEmb_len, -1)])
        
        #### Y
        self.inSampleY = self.data.ts[range(self.m, self.inSampleEmb_len + self.m)]

        self.inSampleY_mean=self.inSampleY.mean(axis=0)
        self.inSampleY_std=self.inSampleY.std(axis=0)

        self.inSampleY = (self.inSampleY-self.inSampleY_mean)/self.inSampleY_std
        
    def standardize_in_sample_nn(self, is_validation = False):
        """
        This is the function of in-sample standardization of nearest neighbour filter implementation
        
        Finished && unchecked
        """
        if(is_validation):
            self.inSampleEmb_len = self.index.validate_start - self.m
        else:
            self.inSampleEmb_len = self.index.test_start - self.m
            
        #### X
        self.inSampleX_nn = np.repeat(np.nan, self.inSampleEmb_len * self.m * self.numLocs * self.numLocs).reshape(self.inSampleEmb_len, self.m, self.numLocs, -1)
            
        # Start the pop-up of X
        for _ in range(self.inSampleEmb_len):
            self.inSampleX_nn[_, ] = np.tile(self.data.ts[range(_, (self.m + _))], self.numLocs).reshape(self.m, self.numLocs, -1)

        self.inSampleX_nn_mean = self.inSampleX_nn.mean(axis=0)
        self.inSampleX_nn_std = self.inSampleX_nn.std(axis=0)
        
        self.inSampleX_nn = (self.inSampleX_nn - self.inSampleX_nn_mean) / self.inSampleX_nn_std
        # End the pop-up of X
        self.inSampleDesignMatrix_nn = np.column_stack([np.repeat(1, self.inSampleEmb_len), self.inSampleX_nn.reshape(self.inSampleEmb_len, -1)])
        
        #### Y
        self.inSampleY = self.data.ts[range(self.m, self.inSampleEmb_len + self.m)]

        self.inSampleY_mean=self.inSampleY.mean(axis=0)
        self.inSampleY_std=self.inSampleY.std(axis=0)

        self.inSampleY = (self.inSampleY - self.inSampleY_mean) / self.inSampleY_std

    def standardize_out_sample(self, is_validation = False):
        if(is_validation):
            self.outSampleEmb_index = np.arange(self.index.validate_start, self.index.validate_end+1)
        else:
            self.outSampleEmb_index = np.arange(self.index.test_start, self.index.test_end+1)
            
        self.outSampleEmb_len = len(self.outSampleEmb_index)
        
        #### X
        self.outSampleX = np.zeros((self.outSampleEmb_len, self.m, self.numLocs)) * np.nan
        for i, ind in enumerate(self.outSampleEmb_index):
            self.outSampleX[i,] = self.data.ts[range(ind - self.m, ind)]
        self.outSampleX = (self.outSampleX - self.inSampleX_mean)/self.inSampleX_std
        self.outSampleDesignMatrix=np.column_stack([np.repeat(1,self.outSampleEmb_len),self.outSampleX.reshape(self.outSampleEmb_len,-1)])

        #### Y
        self.outSampleY = (self.data.ts[self.outSampleEmb_index] - self.inSampleY_mean)/self.inSampleY_std

    def standardize_out_sample_nn(self, is_validation = False):
        if(is_validation):
            self.outSampleEmb_index = np.arange(self.index.validate_start, self.index.validate_end + 1)
        else:
            self.outSampleEmb_index = np.arange(self.index.test_start, self.index.test_end + 1)
            
        self.outSampleEmb_len = len(self.outSampleEmb_index)
        
        #### X
        self.outSampleX_nn = np.zeros((self.outSampleEmb_len, self.m, self.numLocs, self.numLocs)) * np.nan

        # Start the pop-up of X
        for _, ind in enumerate(self.outSampleEmb_index):
            self.outSampleX_nn[_,] = np.tile(self.data.ts[range(ind - self.m, ind)], self.numLocs).reshape(self.m, self.numLocs, -1)
        
        self.outSampleX_nn = (self.outSampleX_nn - self.inSampleX_nn_mean) / self.inSampleX_nn_std
        self.outSampleDesignMatrix_nn = np.column_stack([np.repeat(1, self.outSampleEmb_len), self.outSampleX_nn.reshape(self.outSampleEmb_len, -1)])
        # End the pop-up of X

        #### Y
        self.outSampleY_nn = (self.data.ts[self.outSampleEmb_index] - self.inSampleY_mean) / self.inSampleY_std

    def get_w_and_u(self):
        wMat = np.random.uniform(-self.wWidth,self.wWidth,self.nh*self.nh).reshape(self.nh,-1)
        uMat = np.random.uniform(-self.uWidth,self.uWidth,self.nh*self.nColsU).reshape(self.nColsU,-1)

        #Make W Matrix Sparse 
        for i in range(self.nh):
            numReset=self.nh-np.random.binomial(self.nh,self.wSparsity)
            resetIndex = np.random.choice(self.nh, numReset, replace = False)
            wMat[resetIndex,i]=0

        #Make U Matrix Sparse 
        for i in range(self.nColsU):
            numReset=self.nh-np.random.binomial(self.nh,self.uSparsity)
            resetIndex = np.random.choice(self.nh, numReset, replace = False)
            uMat[i,resetIndex]=0

        #Scale W Matrix
        v = eigh(wMat,eigvals_only=True)
        spectralRadius = max(abs(v))
        wMatScaled=wMat*self.delta/spectralRadius
        
        return wMatScaled, uMat
    
    # Start the ESN-NNF
    def get_wtd(self):
        wMat = np.random.uniform(-self.wWidth, self.wWidth, self.nh * self.nh).reshape(self.nh, -1)

        #Make W Matrix Sparse 
        for i in range(self.nh):
            numReset = self.nh - np.random.binomial(self.nh, self.wSparsity)
            resetIndex = np.random.choice(self.nh, numReset, replace = False)
            wMat[resetIndex, i] = 0

        #Scale W Matrix
        eigenvec = eigh(wMat, eigvals_only=True)
        spectralRadius = max(abs(eigenvec))
        zero_hf = np.zeros((self.nh, self.nf))
        zero_fh = np.zeros((self.nf, self.nh))
        zero_ff = np.zeros((self.nf, self.nf))
        flag_1 = np.concatenate((wMat, zero_hf), axis=-1)
        flag_2 = np.concatenate((zero_fh, zero_ff), axis=-1)
        wMat_new = np.concatenate((flag_1, flag_2), axis=0)
        wMatScaled = wMat_new * self.delta / spectralRadius
        
        return wMatScaled
    
    # Start the ESN-NNF
    def get_utd(self):
        """
        Generate U tilde matrix.
        """

        uMat = np.random.uniform(-self.uWidth, self.uWidth, self.nh * self.nColsU).reshape(self.nColsU, -1)

        #Make U Matrix Sparse 
        for i in range(self.nColsU):
            numReset = self.nh - np.random.binomial(self.nh, self.uSparsity)
            resetIndex = np.random.choice(self.nh, numReset, replace = False)
            uMat[i, resetIndex] = 0

        return uMat
    
    def get_hMat(self,wMat,uMat):
        #Create H Matrix in-sample
        hMatDim = 2*self.nh
        uProdMat=self.inSampleDesignMatrix.dot(uMat);

        hMat = np.zeros((hMatDim,self.inSampleEmb_len))

        xTemp = uProdMat[0,:]
        xTemp = np.tanh(xTemp)

        hMat[0:self.nh,0] = xTemp
        hMat[self.nh:,0] = xTemp*xTemp

        for t in range(1,self.inSampleEmb_len):
            xTemp = wMat.dot(xTemp)+uProdMat[t,:]
            xTemp = np.tanh(xTemp)

            hMat[0:self.nh,t] = xTemp*self.alpha + hMat[0:self.nh,t-1]*(1-self.alpha)
            hMat[self.nh:,t] = hMat[0:self.nh,t]*hMat[0:self.nh,t]

        #Create H Matrix out-sample
        uProdMatOutSample = self.outSampleDesignMatrix.dot(uMat)
        hMatOutSample = np.zeros((self.outSampleEmb_len,hMatDim))

        xTemp = wMat.dot(xTemp)+uProdMatOutSample[0,:]
        xTemp = np.tanh(xTemp)

        hMatOutSample[0,0:self.nh] = xTemp
        hMatOutSample[0,self.nh:] = xTemp*xTemp

        for t in range(1,self.outSampleEmb_len):
            xTemp = wMat.dot(xTemp)+uProdMatOutSample[t,:]
            xTemp = np.tanh(xTemp)

            hMatOutSample[t,0:self.nh] = xTemp*self.alpha + hMatOutSample[t-1,0:self.nh]*(1-self.alpha)
            hMatOutSample[t,self.nh:] = hMatOutSample[t,0:self.nh]*hMatOutSample[t,0:self.nh]

        return hMat, hMatOutSample
    
    def get_hMat_nn(self,wMat,uMat):
        """
        Nearest neighbor version.
        """

        #Create H Matrix in-sample
        hMatDim = 2*self.nh
        uProdMat=self.inSampleDesignMatrix.dot(uMat);

        hMat = np.zeros((hMatDim,self.inSampleEmb_len))

        xTemp = uProdMat[0,:]
        xTemp = np.tanh(xTemp)

        hMat[0:self.nh,0] = xTemp
        hMat[self.nh:,0] = xTemp*xTemp

        for t in range(1,self.inSampleEmb_len):
            xTemp = wMat.dot(xTemp)+uProdMat[t,:]
            xTemp = np.tanh(xTemp)

            hMat[0:self.nh,t] = xTemp*self.alpha + hMat[0:self.nh,t-1]*(1-self.alpha)
            hMat[self.nh:,t] = hMat[0:self.nh,t]*hMat[0:self.nh,t]

        #Create H Matrix out-sample
        uProdMatOutSample = self.outSampleDesignMatrix.dot(uMat)
        hMatOutSample = np.zeros((self.outSampleEmb_len,hMatDim))

        xTemp = wMat.dot(xTemp)+uProdMatOutSample[0,:]
        xTemp = np.tanh(xTemp)

        hMatOutSample[0,0:self.nh] = xTemp
        hMatOutSample[0,self.nh:] = xTemp*xTemp

        for t in range(1,self.outSampleEmb_len):
            xTemp = wMat.dot(xTemp)+uProdMatOutSample[t,:]
            xTemp = np.tanh(xTemp)

            hMatOutSample[t,0:self.nh] = xTemp*self.alpha + hMatOutSample[t-1,0:self.nh]*(1-self.alpha)
            hMatOutSample[t,self.nh:] = hMatOutSample[t,0:self.nh]*hMatOutSample[t,0:self.nh]

        return hMat, hMatOutSample

    def train(self,hyper_para):
        
        self.m, self.nh, self.ridge, self.delta, self.alpha, self.wWidth, self.uWidth, self.wSparsity, self.uSparsity = hyper_para; 
        
        self.m = int(self.m)
        self.nh = int(self.nh)
        
        self.numLocs = self.data.ts.shape[1]
        
        self.standardize_in_sample()
        
    def train_nn(self,hyper_para):
        
        self.m, self.nh, self.ridge, self.delta, self.alpha, self.wWidth, self.uWidth, self.wSparsity, self.uSparsity, self.nf, self.k = hyper_para; 
        
        self.m = int(self.m)
        self.nh = int(self.nh)
        
        self.numLocs = self.data.ts.shape[1]
        
        self.standardize_in_sample()
            
    def forecast(self):
        '''
            Forecast.
        
            Obtain the forecast matrix, forMat: 
                * dimension: (#ensemble, #forecast time points, #locations, #prediction ahead time)
                * forMat[e,t,s,p] is the (p+1)-time ahead forecast for time t (instead of t+p+1!!) 
                    at location s from e-th ensemble
        '''
        print("Forecasting, ensemble: ", end="")

        self.standardize_out_sample()
        self.forMat = np.ones((self.ensembleLen,self.outSampleEmb_len,self.numLocs,self.numTimePred)) * np.nan
        self.nColsU = self.numLocs * self.m + 1

        for iEnsem in range(self.ensembleLen):
            print(iEnsem+1,end=" ")

            wMat, uMat = self.get_w_and_u()

            hMat, hMatOutSample = self.get_hMat(wMat,uMat)
            
            #Ridge Regression to get out-sample forecast
            tmp = hMat.dot(hMat.transpose())
            np.fill_diagonal(tmp,tmp.diagonal()+self.ridge)

            self.forMat[iEnsem,:,:,0] = hMatOutSample.dot(np.linalg.solve(tmp,hMat.dot(self.inSampleY)))

            #Transform to the original scale
            self.forMat[iEnsem,:,:,0] = self.forMat[iEnsem,:,:,0] * self.inSampleY_std + self.inSampleY_mean
                
            #### Prediction at t + pred_lag + 1 where t is the current time
            hMatDim = 2*self.nh
            for pred_lag in range(1,self.numTimePred):

                #Create H Matrix out-sample for prediction more than one lead time
                self.outSampleX_mixed = self.outSampleX.copy()

                for i in range(min(pred_lag,self.m)):
                    ii = i+1
                    self.outSampleX_mixed[pred_lag:,-ii,:] = (self.forMat[iEnsem,(pred_lag-ii):(-ii),:,pred_lag-ii] - 
                                                              self.inSampleX_mean[-ii])/self.inSampleX_std[-ii]

                self.outSampleX_mixed[0:pred_lag,] = np.nan
                self.outSampleDesignMatrix_mixed = np.column_stack([np.repeat(1,self.outSampleEmb_len),
                                                    self.outSampleX_mixed.reshape(self.outSampleEmb_len,-1)])
    
                uProdMatOutSample = self.outSampleDesignMatrix_mixed.dot(uMat)

                hMatOutSample_new = np.zeros((self.outSampleEmb_len,hMatDim)) * np.nan

                xTemp = hMatOutSample[pred_lag-1,0:self.nh]
                xTemp = wMat.dot(xTemp)+uProdMatOutSample[pred_lag,:]
                xTemp = np.tanh(xTemp)

                hMatOutSample_new[pred_lag,0:self.nh] = xTemp
                hMatOutSample_new[pred_lag,self.nh:] = xTemp*xTemp

                for t in range(pred_lag+1,self.outSampleEmb_len):
                    xTemp = hMatOutSample[t-1,0:self.nh]
                    xTemp = wMat.dot(xTemp)+uProdMatOutSample[t,:]
                    xTemp = np.tanh(xTemp)

                    hMatOutSample_new[t,0:self.nh] = xTemp*self.alpha + hMatOutSample_new[t-1,0:self.nh]*(1-self.alpha)
                    hMatOutSample_new[t,self.nh:] = hMatOutSample_new[t,0:self.nh] * hMatOutSample_new[t,0:self.nh]

                hMatOutSample = hMatOutSample_new.copy()
                
                self.forMat[iEnsem,:,:,pred_lag] = hMatOutSample.dot(np.linalg.solve(tmp,hMat.dot(self.inSampleY)))
                
                #Transform to the original scale
                self.forMat[iEnsem,:,:,pred_lag] = self.forMat[iEnsem,:,:,pred_lag] * self.inSampleY_std + self.inSampleY_mean

    def cross_validation(self,cv_para,mChanged = True):
        '''
            Input: 
                cv_para: the cross-validation parameter [m, nh, ridge, delta, alpha, wWidth, uWidth, wSparsity, uSparsity]
                mChange: if m in this cross-validation is different than the last one. If no, there is no need to 
                        re-standardize the in-sample and out-sample data
            Output:
                MSE: vector of MSE with dimension self.numTimePred, which are the mean forecast square error for the different
                     time ahead forecast
        '''
    
        print("Cross Validation with Multiple Lead Times:")

        self.numLocs = self.data.ts.shape[1]
        
        self.m, self.nh, self.ridge, self.delta, self.alpha, self.wWidth, self.uWidth, self.wSparsity, self.uSparsity = cv_para; 
    
        self.m = int(self.m)
        self.nh = int(self.nh)
        
        self.nColsU = self.numLocs * self.m + 1
        
        if(mChanged):
            self.standardize_in_sample(True)
            self.standardize_out_sample(True)
        
        forMatCV = np.zeros((self.ensembleLen,self.outSampleEmb_len,self.numLocs,self.numTimePred))

        for iEnsem in range(self.ensembleLen):
            wMat, uMat = self.get_w_and_u();

            hMat, hMatOutSample = self.get_hMat(wMat,uMat);

            #Ridge Regression to get out-sample forecast
            tmp = hMat.dot(hMat.transpose())
            np.fill_diagonal(tmp,tmp.diagonal()+self.ridge)

            forMatCV[iEnsem,:,:,0] += hMatOutSample.dot(np.linalg.solve(tmp,hMat.dot(self.inSampleY)))
            
            #### Prediction at t + pred_lag + 1 where t is the current time
            hMatDim = 2*self.nh
            for pred_lag in range(1,self.numTimePred):

                #Create H Matrix out-sample for prediction more than one lead time
                outSampleX_mixed = self.outSampleX.copy()

                for i in range(min(pred_lag,self.m)):
                    ii = i+1
                    forMatCV_scaled_back  = forMatCV[iEnsem,(pred_lag-ii):(-ii),:,pred_lag-ii] * self.inSampleY_std + self.inSampleY_mean
                    outSampleX_mixed[pred_lag:,-ii,:] = (forMatCV_scaled_back - self.inSampleX_mean[-ii])/self.inSampleX_std[-ii]

                outSampleX_mixed[0:pred_lag,] = np.nan
                outSampleDesignMatrix_mixed = np.column_stack([np.repeat(1,self.outSampleEmb_len),
                                                    outSampleX_mixed.reshape(self.outSampleEmb_len,-1)])
    
                uProdMatOutSample = outSampleDesignMatrix_mixed.dot(uMat)

                hMatOutSample_new = np.zeros((self.outSampleEmb_len,hMatDim)) * np.nan

                xTemp = hMatOutSample[pred_lag-1,0:self.nh]
                xTemp = wMat.dot(xTemp)+uProdMatOutSample[pred_lag,:]
                xTemp = np.tanh(xTemp)

                hMatOutSample_new[pred_lag,0:self.nh] = xTemp
                hMatOutSample_new[pred_lag,self.nh:] = xTemp*xTemp

                for t in range(pred_lag+1,self.outSampleEmb_len):
                    xTemp = hMatOutSample[t-1,0:self.nh]
                    xTemp = wMat.dot(xTemp)+uProdMatOutSample[t,:]
                    xTemp = np.tanh(xTemp)

                    hMatOutSample_new[t,0:self.nh] = xTemp*self.alpha + hMatOutSample_new[t-1,0:self.nh]*(1-self.alpha)
                    hMatOutSample_new[t,self.nh:] = hMatOutSample_new[t,0:self.nh] * hMatOutSample_new[t,0:self.nh]

                hMatOutSample = hMatOutSample_new.copy()
                
                forMatCV[iEnsem,:,:,pred_lag] = hMatOutSample.dot(np.linalg.solve(tmp,hMat.dot(self.inSampleY)))
        
        
        forMatCVmean = forMatCV.mean(axis = 0)

        diff = np.ndarray(shape = forMatCVmean.shape) * np.nan

        for i in range(self.numTimePred):
            diff[:,:,i] = forMatCVmean[:,:,i] - self.outSampleY

        MSPE = np.nanmean(diff**2,axis=(0,1))
        
        return MSPE
    
    def compute_forecast_mean(self):
        '''
            Compute the ensemble forecast mean, forMean:
                * dimension: (#forecast time points, #locations, #prediction ahead time)
                * forMean[t,s,p] is the (p+1)-time ahead forecast mean for time t (instead of t+p+1!!) at location s 
        '''
        self.forMean = self.forMat.mean(axis=0)
        self.forMeanComputed = True
        
    def compute_forecast_error(self):
        '''
            Compute the error by the ensemble forecast mean, forError:
                * dimension: (#forecast time points, #locations, #prediction ahead time)
                * forError[t,s,p] is the (p+1)-time ahead forecast error for time t (instead of t+p+1!!) at location s 
        '''
        if(not self.forMeanComputed):
            self.compute_forecast_mean()

        self.forError = np.zeros_like(self.forMean)
        self.forError.fill(np.nan)
        
        for ahead in range(self.numTimePred):
            self.forError[:,:,ahead] = self.forMean[:,:,ahead] -  self.data.ts[self.outSampleEmb_index]

        self.forErrorComputed = True

    def compute_MSPE(self):
        if(not self.forErrorComputed):
            self.compute_forecast_error()

        return np.nanmean(self.forError**2,axis = (0,1))