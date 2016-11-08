import h5py
import numpy as np
from keras.utils import np_utils

def LoadData(filename,FractionTest=.1,MaxEvents=-1,Classification=True):

    F=h5py.File(filename,"r")

    X_In_Shape=shape=F["images"].shape
    N=X_In_Shape[0]
    
    if MaxEvents>0:
        X_In=F["images"][:MaxEvents]
        Y_In=F["OneHot"][:MaxEvents]
        YT_In=F["Index"][:MaxEvents]
        N=MaxEvents
    else:
        X_In=F["images"]
        Y_In=F["OneHot"]
        YT_In=F["Index"]

    N_Test=int(round(FractionTest*N))
    N_Train=N-N_Test
        
    Train_X=X_In[:N_Train]
    Train_Y=Y_In[:N_Train]
    Train_TY=YT_In[:N_Train]

    Test_X=X_In[N_Train:]
    Test_Y=Y_In[N_Train:]
    Test_TY=YT_In[N_Train:]

    if Classification:
        Test_Y=np.sum(Test_Y.reshape(N_Test,2,100),axis=2)
        Train_Y=np.sum(Train_Y.reshape(N_Train,2,100),axis=2)
        
    return (Train_X, Train_Y,  Train_TY), (Test_X, Test_Y, Test_TY)


if __name__ == '__main__':
    (x,y,z),(xx,yy,zz)=LoadData("/home/afarbin/LCD/DLTools/LCD-Electrons-Pi0.h5",.1,100)


