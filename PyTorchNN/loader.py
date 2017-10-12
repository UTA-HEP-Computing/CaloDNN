# HDF5Dataset is of class torch.utils.data.Dataset, and is initialized with a set of data files and the number of events per file.
# __len__ returns the number of items in the dataset, which is simply the number of files times the number of events per file.
# __getitem__ takes an index and returns that event. First it sees which file the indexed event would be in, and loads that file if it is not already in memory. It reads the entire ECAL, HCAL, and target information of that file into memory. Then it returns info for the requested event.

# OrderedRandomSampler is used to pass indices to HDF5Dataset, but the indices are created in such a way that the first file is completely read first, and then the second file, then the third etc.

import torch.utils.data as data
from torch import from_numpy
import h5py
import numpy as np

def load_hdf5(file):

    """Loads H5 file. Used by HDF5Dataset."""

    with h5py.File(file, 'r') as f:
        ECAL = f['ECAL'][:]
        HCAL = f['HCAL'][:]
        pdgID = f['pdgID'][:]

    return ECAL, HCAL, pdgID

def load_3d_hdf5(file):

    """Loads H5 file and adds an extra dimension for CNN. Used by HDF5Dataset."""

    ECAL, HCAL, pdgID = load_hdf5(file)
    ECAL = np.expand_dims(ECAL, axis=1)
    HCAL = np.expand_dims(HCAL, axis=1)

    return ECAL, HCAL, pdgID

class HDF5Dataset(data.Dataset):

    """Creates a dataset from a set of H5 files.
        Used to create PyTorch DataLoader.
    Arguments:
        dataname_tuples: list of filename tuples, where each tuple will be mixed into a single file
        num_per_file: number of events in each data file
    """

    def __init__(self, dataname_tuples, num_per_file, classPdgID):
        self.dataname_tuples = sorted(dataname_tuples)
        self.num_per_file = num_per_file
        self.fileInMemory = -1
        self.ECAL = None
        self.HCAL = None
        self.classPdgID = {}
        for i, ID in enumerate(classPdgID):
            self.classPdgID[ID] = i

    def __getitem__(self, index):
        fileN = index/self.num_per_file
        indexInFile = index%self.num_per_file
        if(fileN != self.fileInMemory):
            for dataname in dataname_tuples[fileN]:
                file_ECAL, file_HCAL, file_pdgID = load_hdf5(dataname)
                if (self.ECAL != None):
                    self.ECAL = np.append(self.ECAL, file_ECAL)
                    self.HCAL = np.append(self.HCAL, file_HCAL)
                    pdgID = np.append(self.pdgID, file_pdgID) 
                    self.y = [classPdgID[abs(i)] for i in pdgID]
                else:
                    self.ECAL = file_ECAL
                    self.HCAL = file_HCAL
                    self.y = [classPdgID[abs(i)] for i in file_pdgID]
            self.fileInMemory = fileN
        return self.ECAL[indexInFile], self.HCAL[indexInFile], self.pdgID[indexInFile]

    def __len__(self):
        return len(self.dataname_tuples)*self.num_per_file

class OrderedRandomSampler(object):

    """Samples subset of elements randomly, without replacement.
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.data_source = data_source
        self.num_per_file = self.data_source.num_per_file
        self.num_of_files = len(self.data_source.dataname_tuples)

    def __iter__(self):
        indices=np.array([],dtype=np.int64)
        for i in range(self.num_of_files):
            indices=np.append(indices, np.random.permutation(self.num_per_file)+i*self.num_per_file)
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
