import torch.utils.data as data
import torch
import h5py

class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()
        hf = h5py.File(file_path)#'r'
        self.data = hf.get('data')
        self.target = hf.get('label')
        # self.data = self.data.reshape((-1, 25, 25))
        # self.target = self.target.reshape((-1, 25, 25))
    def __getitem__(self, index):        
#        print ('data size:',self.data.shape)
        return torch.from_numpy(self.data[index,:,:,:]).float(), torch.from_numpy(self.target[index,:,:,:]).float()
        
    def __len__(self):

        return self.data.shape[0]
    

