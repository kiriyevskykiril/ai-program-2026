# imports
from typing import Callable, Optional, List, Self
from numpy.typing import NDArray
import numpy as np
import scipy as sp
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score

lKernelType = ['Cosine', 'Gaussian', 'Triangular', 'Uniform']

lH          = list(np.linspace(4, 8, 100))


def CosineKernel( vU: np.ndarray ) -> np.ndarray:
    return (np.abs(vU) < 1) * (1 + np.cos(np.pi * vU))

def GaussianKernel( vU: np.ndarray ) -> np.ndarray:
    return np.exp(-0.5 * np.square(vU))

def TriangularKernel( vU: np.ndarray ) -> np.ndarray:
    return (np.abs(vU) < 1) * (1 - np.abs(vU))

def UniformKernel( vU: np.ndarray ) -> np.ndarray:
    return 1 * (np.abs(vU) < 0.5)

lKernels = [('Cosine', CosineKernel), ('Gaussian', GaussianKernel), ('Triangular', TriangularKernel), ('Uniform', UniformKernel)]

def ApplyKernelRegression(hKernel: Callable[[NDArray], NDArray], paramH: float, mX:NDArray, vY: NDArray, mX0:NDArray, metricType: str = 'euclidean', zeroThr:float=1e-9) -> NDArray:

    if paramH <= 0:
        raise ValueError('paramH must be positive')
    mD = sp.spatial.distance.cdist(mX0, mX, metric=metricType) # matrix of disatnces between every row in matrox mX0 to every row in matrix mX
    mW = hKernel(mD / paramH) # calculation of weights with kernel function and paramH (channel width)
    vK = mW @ vY 
    vW = np.sum(mW, axis=1) # Summation of every row for further normalization
    vI = np.abs(vW) < zeroThr # finding of indexes with values which a close to zero, to evoid devision by zero 
    vK[vI] = 0.0 # asigning of zero to indexes which where very close to zero
    vW[vI] = 1.0 # by assigning of value 1 to vW[vI] we devide 0/1 and get zero and by this way avoiding deviding by zero
    vYPred = vK / vW

    return vYPred

# The Kernel Regressor Class

class KernelRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, kernelType: str = 'Gaussian', paramH: Optional[float] = None, metricType: str = 'euclidean', lKernels: List = lKernels):
        #===========================Fill This===========================#
        # 1. Add `kernelType` as an attribute of the object.
        # 2. Define the kernel from `lKernels` as `self.hKernel`.
        # 3. Add `paramH` as an attribute of the object.

        # !! Verify the input string of the kernel is within `lKernels`.
        self.kernelType = kernelType
        hKernel = None
        for tKernel in lKernels:
            if tKernel[0] == kernelType:
                hKernel = tKernel[1]
                break
        if hKernel is not None:
            self.hKernel = hKernel
        else:
            raise ValueError(f'The kernel in kernelType = {kernelType} is not in lKernels.')
        self.paramH     = paramH
        #===============================================================#
        # We must set all input parameters as attributes
        self.metricType = metricType
        self.lKernels   = lKernels
    
    def fit(self, mX: NDArray, vY: NDArray) -> Self:
        
        if np.ndim(mX) != 2:
            raise ValueError(f'The input `mX` must be an array like of size (n_samples, n_features) !')
        
        if mX.shape[0] !=  vY.shape[0]:
            raise ValueError(f'The input `mX` must be an array like of size (n_samples, n_features) and `vY` must be (n_samples) !')
        
        #===========================Fill This===========================#
        # 1. Extract the number of samples.
        # 2. Set the bandwidth using Silverman's rule of thumb if it is not set (`None`).
        # 3. Keep a copy of `mX` as a reference grid of features `mG`.
        # 4. Keep a copy of `vY` as a reference values.
        numSamples = mX.shape[0]
        if self.paramH is None:
            # Using Silverman's rule of thumb.
            # It is optimized for Density Estimation for Univariate Gaussian like data.
            σ = np.sqrt(np.sum(np.sqaure(mX - np.mean(mX, axis = 0))))
            self.paramH = 1.06 * σ * (numSamples ** (-0.2))
        
        # The data which is the grid of data to interpolate by
        self.mXd = mX.copy() #<! Copy!
        self.vYd = vY.copy() #<! Copy!
        #===============================================================#

        return self
    
    def predict(self, mX: NDArray) -> NDArray:
        # Given the pair `(mXd, vYd)` in `fit()`, calculate the kernel regression over each row of `mX`

        if np.ndim(mX) != 2:
            raise ValueError(f'The input `mX` must be an array like of size (n_samples, n_features) !')

        if mX.shape[1] != self.mXd.shape[1]:
            raise ValueError(f'The input `mX` must be an array like of size (n_samples, n_features) where `n_features` matches the number of feature in `fit()` !')

        return ApplyKernelRegression(self.hKernel, self.paramH, self.mXd, self.vYd, mX, self.metricType)
    
    def score(self, mX: NDArray, vY: NDArray) -> float:
        # Return the R2 as the score

        if (np.size(vY) != np.size(mX, axis = 0)):
            raise ValueError(f'The number of samples in `mX` must match the number of labels in `vY`.')

        #===========================Fill This===========================#
        # 1. Apply the prediction on the input features.
        # 2. Calculate the R2 score (You may use `r2_score()`).
        vYPred  = self.predict(mX)
        valR2   = r2_score(vY, vYPred)
        #===============================================================#

        return valR2    

