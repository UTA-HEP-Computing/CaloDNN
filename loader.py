import torch.utils.data as data
from torch import from_numpy
import h5py
import numpy as np

def load_hdf5(file):

    """Loads H5 file. Used by HDF5Dataset."""

    with h5py.File(file, 'r') as f:
        ECAL = f['ECAL'][:]
        HCAL = f['HCAL'][:]
        data = [ECAL,HCAL]
        target = f['target'][:]

    return data, target

def load_3d_hdf5(file):

    """Loads H5 file and adds an extra dimension for CNN. Used by HDF5Dataset."""

    data, target = load_hdf5(file)
    data = np.expand_dims(data, axis=1)

    return data, target

class HDF5Dataset(data.Dataset):

    """Creates a dataset from a set of H5 files.
        Changes self.data and self.target based on file number of queried index.
        Used to create PyTorch DataLoader.
    Arguments:
        data_files: list of file names
        num_per_file: number of events in each data file
    """

    def __init__(self, data_files, num_per_file):
        self.data_files = sorted(data_files)
        self.num_per_file = num_per_file
        self.fp = -1
        self.data = None
        self.target = None

    def __getitem__(self, index):
        if(index/self.num_per_file != self.fp):
            self.data, self.target = load_hdf5(self.data_files[index/self.num_per_file])
            self.fp=index/self.num_per_file
        return [self.data[0][index%self.num_per_file], self.data[1][index%self.num_per_file]], self.target[index%self.num_per_file]

    def __len__(self):
        return len(self.data_files)*self.num_per_file

class OrderedRandomSampler(object):

    """Samples subset of elements randomly, without replacement.
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.data_source = data_source
        self.num_per_file = self.data_source.num_per_file
        self.num_of_files = len(self.data_source.data_files)

    def __iter__(self):
        indices=np.array([],dtype=np.int64)
        for i in range(self.num_of_files):
            indices=np.append(indices, np.random.permutation(self.num_per_file)+i*self.num_per_file)
	    print indices
        return iter(from_numpy(indices))

    def __len__(self):
        return len(self.data_source)

# def loadFiles():

    # particle_name_0 = "Ele"
    # particle_name_1 = "ChPi"
    # mixed_particle_name = "EleChPi"
    # samplePath = "~/Projects/DNNCalorimeter/Data/V2/MixedEleChPi/"

    # nworkers = 0
    # batch_size = 1000

    # ###############
    # # Mixed files #
    # ###############

    # train_files = samplePath+"Train/"+mixed_particle_name+"_*.h5"
    # test_files = samplePath+"Test/"+mixed_particle_name+"_*.h5"
    # # val_files = samplePath+"Test/"+mixed_particle_name+"_0.h5"
    # eventsPerFile = 20000

    # train_files = glob.glob(train_files)
    # test_files = glob.glob(test_files)
    # # val_files = glob.glob(val_files)
    # train_set = HDF5Dataset(train_files,eventsPerFile)
    # test_set = HDF5Dataset(test_files,eventsPerFile)
    # # val_set = HDF5Dataset(val_files,eventsPerFile)
    # train_loader = data.DataLoader(dataset=train_set,batch_size=batch_size,sampler=OrderedRandomSampler(train_set),num_workers=nworkers)
    # test_loader = data.DataLoader(dataset=test_set,batch_size=batch_size,sampler=OrderedRandomSampler(test_set),num_workers=nworkers)
    # # val_loader = data.DataLoader(dataset=val_set,batch_size=batch_size,sampler=OrderedRandomSampler(val_set),num_workers=nworkers)

    #########################
    # Single particle files #
    #########################

    # train_ele_files = "/data/LCD/V2/SimpleDownsampled"+mixed_particle_name+"_1_merge/"+particle_name_0+"Train/"+particle_name_0+"_*.h5"
    # test_ele_files = "/data/LCD/V2/SimpleDownsampled"+mixed_particle_name+"_1_merge/"+particle_name_0+"Test/"+particle_name_0+"_*.h5"
    # train_chpi_files = "/data/LCD/V2/SimpleDownsampled"+mixed_particle_name+"_1_merge/"+particle_name_1+"Train/"+particle_name_1+"_*.h5"
    # test_chpi_files = "/data/LCD/V2/SimpleDownsampled"+mixed_particle_name+"_1_merge/"+particle_name_1+"Test/"+particle_name_1+"_*.h5"

    # train_ele_files = glob.glob(train_ele_files)
    # test_ele_files = glob.glob(test_ele_files)
    # train_chpi_files = glob.glob(train_chpi_files)
    # test_chpi_files = glob.glob(test_chpi_files)
    # train_ele_set = HDF5Dataset(train_ele_files, 10000)
    # test_ele_set = HDF5Dataset(test_ele_files, 10000)
    # train_chpi_set = HDF5Dataset(train_chpi_files, 10000)
    # test_chpi_set = HDF5Dataset(test_chpi_files, 10000)
    # train_ele_loader = data.DataLoader(dataset=train_ele_set,batch_size=batch_size,sampler=OrderedRandomSampler(train_ele_set),num_workers=nworkers)
    # test_ele_loader = data.DataLoader(dataset=test_ele_set,batch_size=batch_size,sampler=OrderedRandomSampler(test_ele_set),num_workers=nworkers)
    # train_chpi_loader = data.DataLoader(dataset=train_chpi_set,batch_size=batch_size,sampler=OrderedRandomSampler(train_chpi_set),num_workers=nworkers)
    # test_chpi_loader = data.DataLoader(dataset=test_chpi_set,batch_size=batch_size,sampler=OrderedRandomSampler(test_chpi_set),num_workers=nworkers)
