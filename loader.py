import torch.utils.data as data
from torch import from_numpy
import h5py
import glob
import numpy as np

class OrderedRandomSampler(object):
    """Samples subset of elements randomly, without replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        self.num_per_file=self.data_source.num_per_file
        indices=np.array([],dtype=np.int64)
        num_of_files=len(self.data_source)/self.num_per_file
        for i in range(num_of_files):
            indices=np.append(indices, np.random.permutation(self.num_per_file)+i*self.num_per_file)

        return iter(from_numpy(indices))

    def __len__(self):
        return len(self.data_source)

def load_hdf5(file):
    with h5py.File(file, 'r') as f:
        ECAL=f['ECAL'][:]
        HCAL=f['HCAL'][:]
        data=[ECAL,HCAL]
        target=f['target'][:]
        #target=[0]

    return data, target

def load_3d_hdf5(file):
    with h5py.File(file, 'r') as f:
        ECAL=np.expand_dims(f['ECAL'][:], axis=1)
        HCAL=np.expand_dims(f['HCAL'][:], axis=1)
        data=[ECAL,HCAL]
        target=f['target'][:]

    return data, target

class HDF5Dataset(data.Dataset):
    def __init__(self, data_files, num_per_file=20000):
        self.data_files = sorted(data_files)
        self.num_per_file=num_per_file
        self.fp=-1
        self.data=None
        self.target=None

    def __getitem__(self, index):
        if(index/self.num_per_file != self.fp):
            self.data, self.target = load_hdf5(self.data_files[index/self.num_per_file])
            self.fp=index/self.num_per_file
        return [self.data[0][index%self.num_per_file], self.data[1][index%self.num_per_file]], self.target[index%self.num_per_file]

    def __len__(self):
        return len(self.data_files)*self.num_per_file


particle_name_0 = "Gamma"
particle_name_1 = "Pi0"
mixed_particle_name = "GammaPi0"

train_files="/data/LCD/V2/MixedDownsampled"+mixed_particle_name+"Train_1_merged/"+mixed_particle_name+"_*.h5"
test_files="/data/LCD/V2/MixedDownsampled"+mixed_particle_name+"Test_1_merged/"+mixed_particle_name+"_*.h5"
val_files="/data/LCD/V2/MixedDownsampled"+mixed_particle_name+"Test_1_merged/"+mixed_particle_name+"_0.h5"
train_files=glob.glob(train_files)
test_files=glob.glob(test_files)
val_files=glob.glob(val_files)

nworkers=0
train_batch_size=1000
test_batch_size=1000
val_batch_size=1000

train_set = HDF5Dataset(train_files)
test_set = HDF5Dataset(test_files)
val_set = HDF5Dataset(val_files)
train_loader = data.DataLoader(dataset=train_set, batch_size=train_batch_size,sampler=OrderedRandomSampler(train_set), num_workers=nworkers)
test_loader = data.DataLoader(dataset=test_set, batch_size=test_batch_size,sampler=OrderedRandomSampler(test_set), num_workers=nworkers)
val_loader = data.DataLoader(dataset=val_set, batch_size=val_batch_size,sampler=OrderedRandomSampler(val_set), num_workers=nworkers)


train_ele_files="/data/LCD/V2/SimpleDownsampled"+mixed_particle_name+"_1_merge/"+particle_name_0+"Train/"+particle_name_0+"_*.h5"
test_ele_files="/data/LCD/V2/SimpleDownsampled"+mixed_particle_name+"_1_merge/"+particle_name_0+"Test/"+particle_name_0+"_*.h5"

train_chpi_files="/data/LCD/V2/SimpleDownsampled"+mixed_particle_name+"_1_merge/"+particle_name_1+"Train/"+particle_name_1+"_*.h5"
test_chpi_files="/data/LCD/V2/SimpleDownsampled"+mixed_particle_name+"_1_merge/"+particle_name_1+"Test/"+particle_name_1+"_*.h5"

train_ele_files=glob.glob(train_ele_files)
test_ele_files=glob.glob(test_ele_files)

train_chpi_files=glob.glob(train_chpi_files)
test_chpi_files=glob.glob(test_chpi_files)

train_ele_set = HDF5Dataset(train_ele_files, 10000)
test_ele_set = HDF5Dataset(test_ele_files, 10000)

train_chpi_set = HDF5Dataset(train_chpi_files, 10000)
test_chpi_set = HDF5Dataset(test_chpi_files, 10000)

batch_size = 1000

train_ele_loader = data.DataLoader(dataset=train_ele_set, batch_size=batch_size,sampler=OrderedRandomSampler(train_ele_set), num_workers=nworkers)
test_ele_loader = data.DataLoader(dataset=test_ele_set, batch_size=batch_size,sampler=OrderedRandomSampler(test_ele_set), num_workers=nworkers)

train_chpi_loader = data.DataLoader(dataset=train_chpi_set, batch_size=batch_size,sampler=OrderedRandomSampler(train_chpi_set), num_workers=nworkers)
test_chpi_loader = data.DataLoader(dataset=test_chpi_set, batch_size=batch_size,sampler=OrderedRandomSampler(test_chpi_set), num_workers=nworkers)
