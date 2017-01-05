import h5py
import numpy as np
from keras.utils import np_utils

def LoadData(filename, FractionTest=.1, MaxEvents=-1, Classification=True):

    F = h5py.File(filename,"r")

    X_In_Shape = F["images"].shape
    N = X_In_Shape[0]
    
    if MaxEvents>0:
        X_In = F["images"][:MaxEvents]
        Y_In = F["OneHot"][:MaxEvents]
        YT_In = F["Index"][:MaxEvents]
        N=MaxEvents
    else:
        X_In = F["images"]
        Y_In = F["OneHot"]
        YT_In = F["Index"]

    N_Test = int(round(FractionTest*N))
    N_Train = N-N_Test
        
    Train_X = X_In[:N_Train]
    Train_Y = Y_In[:N_Train]
    Train_TY = YT_In[:N_Train]

    Test_X = X_In[N_Train:]
    Test_Y = Y_In[N_Train:]
    Test_TY = YT_In[N_Train:]

    if Classification:
        Test_Y = np.sum(Test_Y.reshape(N_Test,2,100),axis=2)
        Train_Y = np.sum(Train_Y.reshape(N_Train,2,100),axis=2)
        
    return (Train_X, Train_Y,  Train_TY), (Test_X, Test_Y, Test_TY)

def LoadDataGen(filename, Classification=True, BatchSize=1024, Skip=0, Max=-1):

    F = h5py.File(filename,"r")

    X_In_Shape = F["images"].shape
    N = X_In_Shape[0]
    if Max < 0.0 or Max > N:
        Max = N

    while True: 

#        for i_step in xrange(Skip, N-BatchSize, BatchSize):
        for i_step in xrange(Skip, Max, BatchSize):

            X_In = F["images"][i_step:i_step+BatchSize]
            Y_In = F["OneHot"][i_step:i_step+BatchSize]

            if Classification:
                Y_In = np.sum(Y_In.reshape(BatchSize, 2, 100), axis=2)

            # hardcoded normalization
            Norm = 150 # HARDCODED
            X_In = X_In/Norm

            yield X_In, Y_In
            

if __name__ == '__main__':

    InputFile="/scratch/data-backup/afarbin/LCD/LCD-Electrons-Pi0.h5"

#    F = h5py.File(InputFile,"r")

#    (x,y,z),(xx,yy,zz)=LoadData(InputFile,.1,100)
#    (x,y,z),(xx,yy,zz) = LoadDataGen("/scratch/data-backup/afarbin/LCD/LCD-Electrons-Pi0.h5")



