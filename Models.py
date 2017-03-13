from DLTools.ModelWrapper import *

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers import  BatchNormalization,Dropout,Flatten,Merge
from keras.models import model_from_json

class Fully3DImageClassification(ModelWrapper):
    def __init__(self, Name, input_shape, width=0, depth=0, BatchSize=2048,
                 N_classes=100, init=0, BatchNormalization=False, Dropout=False,
                 Activation='relu',**kwargs):

        super(Fully3DImageClassification, self).__init__(Name,**kwargs)

        self.width=width
        self.depth=depth
        self.input_shape=input_shape
        self.N_classes=N_classes
        self.init=init

        self.Dropout=Dropout

        self.BatchSize=BatchSize
        self.BatchNormalization=BatchNormalization
        self.Activation=Activation
        
        self.MetaData.update({ "width":self.width,
                               "depth":self.depth,
                               "Dropout":self.Dropout,
                               "BatchNormalization":BatchNormalization,
                               "input_shape":self.input_shape,
                               "N_classes":self.N_classes,
                               "init":self.init})
    def Build(self):
        model = Sequential()
        model.add(Flatten(batch_input_shape=self.input_shape))

#        model.add(Dense(self.width,init=self.init))
#        model.add(Activation('relu'))

        for i in xrange(0,self.depth):
            if self.BatchNormalization:
                model.add(BatchNormalization())

            model.add(Dense(self.width,init=self.init))
            model.add(Activation(self.Activation))

            if self.Dropout:
                model.add(Dropout(self.Dropout))

        model.add(Dense(self.N_classes, activation='softmax',init=self.init))

        self.Model=model

class MergerModel(ModelWrapper):
    def __init__(self, Name, Models, N_Classes, init):
        super(MergerModel, self).__init__(Name)
        self.Models=Models
        self.N_Classes=N_Classes
        self.init=init
        
    def Build(self):
        model=Sequential()

        MModels=[]

        for m in self.Models:
            MModels.append(m.Model)

        if len(self.Models)>0:
            print "Merged Models"
            model.add(Merge(MModels,mode='concat'))
        model.add(Dense(self.N_Classes, activation='softmax',init=self.init))

        self.Model=model
        
