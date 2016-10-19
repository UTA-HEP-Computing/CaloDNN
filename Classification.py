from DLTools.ModelWrapper import *

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers import  BatchNormalization,Dropout
from keras.models import model_from_json

class FullyConnectedRegression(ModelWrapper):
    def __init__(self, Name, N_input=0, width=0, depth=0, init=0):

        super(FullyConnectedRegression, self).__init__(Name)

        self.width=width
        self.depth=depth
        self.N_input=N_input
        self.init=init

        self.MetaData.update({ "width":self.width,
                               "depth":self.depth,
                               "N_input":self.N_input,
                               "init":self.init})

    def Build(self):
        model = Sequential()
        model.add(Dense(self.width,input_dim=self.N_input,init=self.init))

        model.add(Activation('tanh'))

        for i in xrange(0,self.depth):
            model.add(BatchNormalization())
            model.add(Dense(self.width,init=self.init))
            model.add(Activation('tanh'))
            model.add(Dropout(0.5))

        model.add(Dense(1,input_dim=self.width))

        self.Model=model

class FullyConnectedClassification(ModelWrapper):
    def __init__(self, Name, N_input=0, width=0, depth=0, N_classes=100, init=0):

        super(FullyConnectedClassification, self).__init__(Name)

        self.width=width
        self.depth=depth
        self.N_input=N_input
        self.N_classes=N_classes
        self.init=init

        self.MetaData.update({ "width":self.width,
                               "depth":self.depth,
                               "N_input":self.N_input,
                               "N_classes":self.N_classes,
                               "init":self.init})

    def Build(self):
        model = Sequential()
        model.add(Dense(self.width,input_dim=self.N_input,init=self.init))

        model.add(Activation('relu'))

        for i in xrange(0,self.depth):
#            model.add(BatchNormalization())
            model.add(Dense(self.width,init=self.init))
            model.add(Activation('relu'))
            model.add(Dropout(0.5))

        model.add(Dense(self.N_classes, activation='softmax'))

        self.Model=model

    def Compile(self, Loss="categorical_crossentropy", Optimizer="rmsprop"):
        self.Model.compile(loss=loss, optimizer=optimizer,metrics=["accuracy"])


class CNN3DClassification(ModelWrapper):
    def __init__(self, Name, N_input=0, width=0, depth=0, N_classes=100, init=0):

        super(FullyConnectedClassification, self).__init__(Name)

        self.width=width
        self.depth=depth
        self.N_input=N_input
        self.N_classes=N_classes
        self.init=init

        self.MetaData.update({ "width":self.width,
                               "depth":self.depth,
                               "N_input":self.N_input,
                               "N_classes":self.N_classes,
                               "init":self.init})

    def Build(self):
        model = Sequential()
        model.add(Dense(self.width,input_dim=self.N_input,init=self.init))

        model.add(Activation('relu'))

        for i in xrange(0,self.depth):
#            model.add(BatchNormalization())
            model.add(Dense(self.width,init=self.init))
            model.add(Activation('relu'))
            model.add(Dropout(0.5))

        model.add(Dense(self.N_classes, activation='softmax'))

        self.Model=model

    def Compile(self, Loss="categorical_crossentropy", Optimizer="rmsprop"):
        self.Model.compile(loss=loss, optimizer=optimizer,metrics=["accuracy"])
        
