from DLTools.ModelWrapper import *

from keras.layers.merge import concatenate
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation
from keras.layers import  BatchNormalization,Dropout,Flatten, Input
from keras.models import model_from_json

class Convolutional3D(ModelWrapper):
    def __init__(self, Name, **kwargs):
        super(Convolutional3D, self).__init__(Name,**kwargs)
        self.filters=10
        self.kernel_size=5
        self.strides=1
        self.padding='valid'
        self.data_format='channels_last'
        self.dilation_rate=1
        self.activation='relu'
        self.kernel_initializer='glorot_uniform'
        self.bias_initializer='zeros'
        self.kernel_regularizer=None
        self.bias_regularizer=None
        self.activity_regularizer=None
        self.kernel_constraint=None
        self.bias_constraint=None
        self.N_classes=2


    def Build(self):
        #model=Sequential()
        #model.add(Conv3D(10,(5,5,5),input_shape=(25,25,25,1)
        #model.add(Activation('relu'))
        #model.add(MaxPooling3D(pool_size=(2, 2)))
        #model.add(Flatten())
        #model.add(Dropout(0.5))
        #model.add(Dense(2))
        #model.add(Activation('sigmoid'))

        #model=Dense(self.N_classes, activation='softmax',kernel_initializer=self.kernel_initializer)(model)
        input=Input(shape=(25,25,25,1))
        modelT=BatchNormalization()(input)
        modelT=Conv3D(self.filters, self.kernel_size, strides=(1, 1, 1), padding='valid', data_format=self.data_format, dilation_rate=(1, 1, 1), activation=self.activation, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(modelT)
        modelT=BatchNormalization()(modelT)
        modelT=AveragePooling3D(pool_size=(5, 5, 5), strides=None, padding='valid', data_format='channels_last')(modelT)

        modelT=Flatten()(modelT)
        modelT=BatchNormalization()(modelT)
        modelT=Dropout(0.5)(modelT)
        modelT=Dense(self.N_classes, activation='softmax',kernel_initializer=self.kernel_initializer)(modelT)

        self.inputT=input
        self.modelT=modelT
        
        self.Model=Model(input,modelT)


class Fully3DImageClassification(ModelWrapper):
    def __init__(self, Name, input_shape, width=0, depth=0, BatchSize=2048,
                 N_classes=100, init=0, BatchNormalization=False, Dropout=False,
                 NoClassificationLayer=False,
                 activation='relu',**kwargs):

        super(Fully3DImageClassification, self).__init__(Name,**kwargs)

        self.width=width
        self.depth=depth
        self.input_shape=input_shape
        self.N_classes=N_classes
        self.init=init

        self.Dropout=Dropout

        self.BatchSize=BatchSize
        self.BatchNormalization=BatchNormalization
        self.Activation=activation
        self.NoClassificationLayer=NoClassificationLayer
        
        self.MetaData.update({ "width":self.width,
                               "depth":self.depth,
                               "Dropout":self.Dropout,
                               "BatchNormalization":BatchNormalization,
                               "input_shape":self.input_shape,
                               "N_classes":self.N_classes,
                               "init":self.init})
    def Build(self):
        input=Input(self.input_shape[1:])
        modelT = Flatten(input_shape=self.input_shape)(input)

#        model.add(Dense(self.width,init=self.init))
        modelT = (Activation('relu')(modelT))

        for i in xrange(0,self.depth):
            if self.BatchNormalization:
                modelT=BatchNormalization()(modelT)

            modelT=Dense(self.width,kernel_initializer=self.init)(modelT)
            modelT=Activation(self.Activation)(modelT)

            if self.Dropout:
                modelT=Dropout(self.Dropout)(modelT)

        if not self.NoClassificationLayer:
            modelT=Dense(self.N_classes, activation='softmax',kernel_initializer=self.init)(modelT)

        self.inputT=input
        self.modelT=modelT
        
        self.Model=Model(input,modelT)

class MergerModel(ModelWrapper):
    def __init__(self, Name, Models, N_Classes, init, **kwargs):
        super(MergerModel, self).__init__(Name,**kwargs)
        self.Models=Models
        self.N_Classes=N_Classes
        self.init=init
        
    def Build(self):

        MModels=[]
        MInputs=[]
        for m in self.Models:
            MModels.append(m.modelT)
            MInputs.append(m.inputT)
        if len(self.Models)>0:
            print "Merged Models"
            modelT=concatenate(MModels)#(modelT)
            
        modelT=Dense(self.N_Classes, activation='softmax',kernel_initializer=self.init)(modelT)
        

        self.modelT=modelT
        
        self.Model=Model(MInputs,modelT)

                
class MergerRegEModel(ModelWrapper):
    def __init__(self, Name, Models, init, **kwargs):
        super(MergerRegEModel, self).__init__(Name,**kwargs)
        self.Models=Models
        self.init=init
        
    def Build(self):

        MModels=[]
        MInputs=[]
        for m in self.Models:
            MModels.append(m.modelT)
            MInputs.append(m.inputT)
        if len(self.Models)>0:
            print "Merged Models"
            modelT=concatenate(MModels)#(modelT)
            
        modelT=Dense(1, activation='tanh',kernel_initializer=self.init)(modelT)
        

        self.modelT=modelT
        
        self.Model=Model(MInputs,modelT)

