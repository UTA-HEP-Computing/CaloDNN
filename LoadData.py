import h5py
import numpy as np
from keras.utils import np_utils



def LoadData(filename,FractionTest=.1,MaxEvents=-1):

    F=h5py.File(filename,"r")

    X_In_Shape=shape=F["images"].shape
    N=X_In_Shape[0]
    
    if MaxEvents>0:
        X_In=F["images"][:MaxEvents]
        Y_In=F["OneHot"][:MaxEvents]
        N=MaxEvents
    else:
        X_In=F["images"]
        Y_In=F["OneHot"]


    N_Test=int(round(FractionTest*N))
    N_Train=N-N_Test
        
    Train_X=X_In[:N_Train]
    Train_Y=Y_In[:N_Train]

    Test_X=X_In[N_Train:]
    Test_Y=Y_In[N_Train:]

        
    return (Train_X, Train_Y), (Test_X, Test_Y)


if __name__ == '__main__':

    (x,y),(xx,yy)=LoadData("/home/afarbin/LCD/DLTools/LCD-Electrons-Pi0.h5",.1,100)


