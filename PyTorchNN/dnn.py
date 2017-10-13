import sys
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import loader
import glob
import os
import numpy as np
import h5py as h5

###############
# Set options #
###############

basePath = "/u/sciteam/zhang10/Projects/DNNCalorimeter/Data/V2/EleChPi/"
samplePath = [basePath + "ChPiEscan/ChPiEscan_*.h5", basePath + "EleEscan/EleEscan_*.h5"]
classPdgID = [211, 11] # absolute IDs corresponding to paths above
eventsPerFile = 10000

trainRatio = 0.9
nEpochs = 5
batchSize = 1000
nworkers = 4

parameters = "_".join(sys.argv[1:])
OutPath = "/u/sciteam/zhang10/Projects/DNNCalorimeter/SubmissionScripts/PyTorchNN/Output/"+parameters+"/"

learningRate = float(sys.argv[1]) # 0.01
decayRate = float(sys.argv[2]) # 0
dropoutProb = float(sys.argv[3]) # 0.3
hiddenLayerNeurons = int(sys.argv[4]) # 32
nHiddenLayers = int(sys.argv[5]) # 5

##############
# Load files #
##############

nParticles = len(samplePath)
particleFiles = [[]] * nParticles
for i, particlePath in enumerate(samplePath):
    particleFiles[i] = glob.glob(particlePath)

filesPerParticle = len(particleFiles[0])
nTrain = int(filesPerParticle * trainRatio)
nTest = filesPerParticle - nTrain
trainFiles = []
testFiles = []
for i in range(filesPerParticle):
    newFiles = []
    for j in range(nParticles):
        newFiles.append(particleFiles[j][i])
    if i < nTrain:
        trainFiles.append(newFiles)
    else:
        testFiles.append(newFiles)
eventsPerFile *= nParticles

trainSet = loader.HDF5Dataset(trainFiles, eventsPerFile, classPdgID)
testSet = loader.HDF5Dataset(testFiles, eventsPerFile, classPdgID)
trainLoader = data.DataLoader(dataset=trainSet,batch_size=batchSize,sampler=loader.OrderedRandomSampler(trainSet),num_workers=nworkers)
testLoader = data.DataLoader(dataset=testSet,batch_size=batchSize,sampler=loader.OrderedRandomSampler(testSet),num_workers=nworkers)

##############
# Create net #
##############

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.input = nn.Linear(25 * 25 * 25 + 5 * 5 * 60, hiddenLayerNeurons)
        self.hidden = nn.Linear(hiddenLayerNeurons, hiddenLayerNeurons)
        self.dropout = nn.Dropout(p = dropoutProb)
        self.output = nn.Linear(hiddenLayerNeurons, 2)
    def forward(self, x1, x2):
        x1 = x1.view(-1, 25 * 25 * 25)
        x2 = x2.view(-1, 5 * 5 * 60)
        x = torch.cat((x1,x2), 1)
        x = F.relu(self.input(x))
        for i in range(nHiddenLayers-1):
            x = F.relu(self.hidden(x))
            x = self.dropout(x)
        x = F.softmax(self.output(x))
        return x

net = Net()
net.cuda()

# optimizer # optimizer = optim.Adadelta(net.parameters(), lr=learningRate, weight_decay=decayRate)
optimizer = optim.Adam(net.parameters(), lr=learningRate, weight_decay=decayRate)
lossFunction = nn.CrossEntropyLoss()

###############
# Train model #
###############

loss_history = []
net.train() # set to training mode
for epoch in range(nEpochs):
    running_loss = 0.0
    for i, data in enumerate(trainLoader, 0):
        ECALs, HCALs, ys = data
        ECALs, HCALs, ys = Variable(ECALs.cuda()), Variable(HCALs.cuda()), Variable(ys.cuda())
        optimizer.zero_grad()
        outputs = net(ECALs, HCALs)
        loss = lossFunction(outputs, ys)
        loss.backward()
        optimizer.step()
        running_loss += loss.data[0]
        if i % 5 == 0:
            print('[%d, %5d] train loss: %.10f' % (epoch + 1, i, running_loss)),
            test_loss = 0.0
            net.eval() # set to evaluation mode (turns off dropout)
            for data in testLoader:
                ECALs, HCALs, ys = data
                ECALs, HCALs, ys = Variable(ECALs.cuda()), Variable(HCALs.cuda()), Variable(ys.cuda())
                outputs = net(ECALs, HCALs)
                loss = lossFunction(outputs, ys)
                test_loss += loss.data[0]
            print(', test loss: %.10f' % (test_loss))
            net.train() # set to training mode
            loss_history.append([epoch + 1, i, running_loss, test_loss])
            running_loss = 0.0

if not os.path.exists(OutPath): os.makedirs(OutPath)
file = h5.File(OutPath+"Results.h5", 'w')
file.create_dataset("loss_history", data=np.array(loss_history))

torch.save(net.state_dict(), OutPath+"SavedModel")

print('Finished Training')

######################
# Analysis and plots #
######################

correct = 0
total = 0
net.eval() # set to evaluation mode (turns off dropout)
for data in testLoader:
    ECALs, HCALs, ys = data
    ECALs, HCALs, ys = Variable(ECALs.cuda()), Variable(HCALs.cuda()), Variable(ys.cuda())
    outputs = net(ECALs, HCALs)
    _, predicted = torch.max(outputs.data, 1)
    total += ys.size(0)
    correct += (predicted == ys.data).sum()

print('Accuracy of the network on test samples: %f %%' % (100 * float(correct) / total))
file.create_dataset("test_accuracy", data=np.array([100*float(correct)/total]))
