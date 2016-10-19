import h5py
import numpy as np
from keras.utils import np_utils



def LoadData(filename,FractionTest=.1,MaxEvents=-1, MinEvents=-1, Shuffle=False, Bin=False, N_Inputs=15):

    F=h5py.File(filename,"r")

    X_In=F["images"]
    X_In_Shape=shape=X_In.shape

    Y_In=F["OneHot"]
    
    N=X_In_Shape[0]
    if N>MaxEvents:
        X_In=X_In[:MaxEvents]
        N=MaxEvents

    N_Test=int(round(FractionTest*N))
    N_Train=N-N_Test

        
    Train_X=X_In[:N_Train]
    Train_Y=Y_In[:N_Train]

    Test_X=X_In[N_Train:]
    Test_Y=Y_In[N_Train:]

        
    return (Train_X, Train_Y), (Test_X, Test_Y)



