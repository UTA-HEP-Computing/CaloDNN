from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import cat
import torch.optim as optim
import torch.utils.data as data
import loader
import glob
import numpy as np
import h5py as h5

###############
# Set options #
###############

basePath = "/u/sciteam/zhang10/Projects/DNNCalorimeter/Data/V2/EleChPi/"
samplePath = [basePath + "ChPiEscan/ChPiEscan_*.h5", basePath + "ChPiEscan/EleEscan_*.h5"]
classPdgID = [211, 11] # absolute IDs corresponding to paths above
eventsPerFile = 10000

OutPath = "Output/PyTorchTest/"
trainRatio = 0.8
learningRate = 0.001
decayRate = 0
nEpochs = 10
batchSize = 1000
nworkers = 0

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
        self.fc1 = nn.Linear(25 * 25 * 25 + 5 * 5 * 60, 32)
        self.fc5 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 2)
    def forward(self, x1, x2):
        x1 = x1.view(-1, 25 * 25 * 25)
        x2 = x2.view(-1, 5 * 5 * 60)
        x = cat((x1,x2), 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc5(x))
        x = self.fc4(x)
        return x

net = Net()
net.cuda()

# optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=decay_rate)

###############
# Train model #
###############

loss_history = []
for epoch in range(nEpochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        ECAL, HCAL, labels = Variable(inputs[0].cuda()), Variable(inputs[1].cuda()), Variable(labels.cuda())
        optimizer.zero_grad()
        outputs = net(ECAL, HCAL)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]
        if i % 20 == 19:
            print('[%d, %5d] train loss: %.10f' % (epoch + 1, i + 1, running_loss)),

	    test_loss = 0.0
	    for i, data in enumerate(test_loader, 0):
		ECAL, HCAL, pdgID = data
		ECAL, HCAL, pdgID = Variable(ECAL.cuda()), Variable(HCAL.cuda()), Variable(pdgID.cuda())
		output = net(ECAL, HCAL)
		loss = criterion(output, (abs(pdgID)==11)) # have electron be class 1
		test_loss += loss.data[0]

	    print(', test loss: %.10f' % (test_loss))
            loss_history.append([epoch + 1, i + 1, running_loss, test_loss])
            running_loss = 0.0

loss_history=np.array(loss_history)

epoch_num="1-10"
with h5.File(OutPath+"loss_history_32-6_"+epoch_num+".h5", 'w') as loss_file:
    loss_file.create_dataset("loss", data=loss_history)

from torch import save
save(net.state_dict(), OutPath+"savedmodel_32-6_lr-"+str(learning_rate)+"_dr-"+str(decay_rate)+"_"+epoch_num)

print('Finished Training')

######################
# Analysis and plots #
######################

from torch import max

correct = 0
total = 0
for data in test_loader:
    ECAL, HCAL, pdgID = data
    ECAL, HCAL, pdgID = Variable(ECAL.cuda()), Variable(HCAL.cuda()), Variable(pdgID.cuda())
    output = net(ECAL, HCAL)
    _, predicted = max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == (pdgID==11)).sum() # comparing to electron as class 1

print('Accuracy of the network on test images: %f %%' % (100 * float(correct) / total))
