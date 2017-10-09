from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import cat
import torch.optim as optim
import torch.utils.data as data
import loader
import glob

# options
OutPath="Downsampled_GammaPi0_1_merged_nn_outputs/"
learning_rate=0.001
decay_rate=0

# define the model
class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
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

net = Net2()
net.cuda()

# optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=decay_rate)

# load data files
particle_name_0 = "Ele"
particle_name_1 = "ChPi"
mixed_particle_name = "EleChPi"
samplePath = "~/Projects/DNNCalorimeter/Data/V2/MixedEleChPi/"
eventsPerFile = 20000
batch_size = 1000
nworkers = 0

train_files = samplePath+"Train/"+mixed_particle_name+"_*.h5"
test_files = samplePath+"Test/"+mixed_particle_name+"_*.h5"
train_files = glob.glob(train_files)
test_files = glob.glob(test_files)
train_set = loader.HDF5Dataset(train_files,eventsPerFile)
test_set = loader.HDF5Dataset(test_files,eventsPerFile)
train_loader = data.DataLoader(dataset=train_set,batch_size=batch_size,sampler=loader.OrderedRandomSampler(train_set),num_workers=nworkers)
test_loader = data.DataLoader(dataset=test_set,batch_size=batch_size,sampler=loader.OrderedRandomSampler(test_set),num_workers=nworkers)

# train model
loss_history = []
for epoch in range(10):
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
            print('[%d, %5d] loss: %.10f' %
                    (epoch + 1, i + 1, running_loss)),


            val_loss = 0.0
            for i, data in enumerate(val_loader, 0):
                inputs, labels = data
                ECAL, HCAL, labels = Variable(inputs[0].cuda()), Variable(inputs[1].cuda()), Variable(labels.cuda())
                outputs = net(ECAL, HCAL)
                loss = criterion(outputs, labels)
                val_loss += loss.data[0]

            print('    val loss: %.10f' %
                    (val_loss))


            loss_history.append([epoch + 1, i + 1, running_loss, val_loss])

            running_loss = 0.0

loss_history=np.array(loss_history)

epoch_num="1-10"
with h5py.File(OutPath+"loss_history_32-6_"+epoch_num+".h5", 'w') as loss_file:
    loss_file.create_dataset("loss", data=loss_history)

from torch import save
save(net.state_dict(), OutPath+"savedmodel_32-6_lr-"+str(learning_rate)+"_dr-"+str(decay_rate)+"_"+epoch_num)

print('Finished Training')

# # Analysis
# from torch import max

# correct = 0
# total = 0
# for data in test_loader:
    # images, labels = data
    # ECAL, HCAL, labels = Variable(images[0].cuda()), Variable(images[1].cuda()), labels.cuda()
    # outputs = net(ECAL, HCAL)
    # _, predicted = max(outputs.data, 1)
    # total += labels.size(0)
    # correct += (predicted == labels).sum()

# print('Accuracy of the network on test images: %f %%' % (
        # 100 * float(correct) / total))


# train_ele_gen = iter(train_ele_loader)
# test_ele_gen = iter(test_ele_loader)

# train_chpi_gen = iter(train_chpi_loader)
# test_chpi_gen = iter(test_chpi_loader)

# from torch import Tensor

# outputs0=Tensor().cuda()
# outputs1=Tensor().cuda()
# outputs2=Tensor().cuda()
# outputs3=Tensor().cuda()


# #  separate outputs for training/testing signal/backroung events. 
# for data in train_ele_gen:
    # images, labels = data
    # outputs0 = cat((outputs0, net(Variable(images[0].cuda()),Variable(images[1].cuda())).data))

# for data in test_ele_gen:
    # images, labels = data
    # outputs1 = cat((outputs1, net(Variable(images[0].cuda()),Variable(images[1].cuda())).data))

# for data in train_chpi_gen:
    # images, labels = data
    # outputs2 = cat((outputs2, net(Variable(images[0].cuda()),Variable(images[1].cuda())).data))

# for data in test_chpi_gen:
    # images, labels = data
    # outputs3 = cat((outputs3, net(Variable(images[0].cuda()),Variable(images[1].cuda())).data))

# with h5py.File(OutPath+"out_32-6_"+epoch_num+"_0.h5", 'w') as o1, h5py.File(OutPath+"out_32-6_"+epoch_num+"_1.h5", 'w') as o2, h5py.File(OutPath+"out_32-6_"+epoch_num+"_2.h5", 'w') as o3, h5py.File(OutPath+"out_32-6_"+epoch_num+"_3.h5", 'w') as o4:
    # o1.create_dataset("output", data=outputs0.cpu().numpy())
    # o2.create_dataset("output", data=outputs1.cpu().numpy())
    # o3.create_dataset("output", data=outputs2.cpu().numpy())
    # o4.create_dataset("output", data=outputs3.cpu().numpy())
