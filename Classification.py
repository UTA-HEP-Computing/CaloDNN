from DLTools.ModelWrapper import *

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers import  BatchNormalization,Dropout,Flatten
from keras.models import model_from_json

class Fully3DImageClassification(ModelWrapper):
    def __init__(self, Name, input_shape, width=0, depth=0, BatchSize=2048, N_classes=100, init=0):

        super(Fully3DImageClassification, self).__init__(Name)

        self.width=width
        self.depth=depth
        self.input_shape=input_shape
        self.N_classes=N_classes
        self.init=init

        self.BatchSize=BatchSize
        
        self.MetaData.update({ "width":self.width,
                               "depth":self.depth,
                               "input_shape":self.input_shape,
                               "N_classes":self.N_classes,
                               "init":self.init})
    def Build(self):
        model = Sequential()
        model.add(Flatten(batch_input_shape=self.input_shape))

#        model.add(Dense(self.width,init=self.init))

        model.add(Activation('relu'))

        for i in xrange(0,self.depth):
#            model.add(BatchNormalization())
            model.add(Dense(self.width,init=self.init))
            model.add(Activation('relu'))
            model.add(Dropout(0.5))

        model.add(Dense(self.N_classes, activation='softmax'))

        self.Model=model

    def Compile(self, Loss="categorical_crossentropy", Optimizer="rmsprop"):
        self.Model.compile(loss=Loss, optimizer=Optimizer,metrics=["accuracy"])
