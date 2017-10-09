import loader
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import cat
import torch.optim as optim
import torch.utils.data as data
import loader
import glob
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
sampler=loader.OrderedRandomSampler(train_set)
iter(sampler)
