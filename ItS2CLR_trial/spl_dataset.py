"""Self-Paced Learning Dataset"""
import torch
import numpy as np
from torch.utils.data import Dataset
import pickle

class BagDataset(Dataset):
    def __init__(self, embeddings_dict):
        self.embeddings_dict = embeddings_dict
        self.bag_names = list(embeddings_dict.keys())
    
    def __len__(self):
        return len(self.bag_names)
    
    def __getitem__(self, index):
        bag_name = self.bag_names[index]
        patches = self.embeddings_dict[bag_name]
        feats = np.array([p[1] for p in patches])
        label = 1 if "tumor" in bag_name.lower() else 0
        return torch.FloatTensor(feats), torch.LongTensor([label]), [p[0] for p in patches], bag_name


class BagDatasetIns(Dataset):
    def __init__(self, embeddings_dict):
        self.embeddings_dict = embeddings_dict
        self.bag_names = list(embeddings_dict.keys())
    
    def __len__(self):
        return len(self.bag_names)
    
    def __getitem__(self, index):
        bag_name = self.bag_names[index]
        patches = self.embeddings_dict[bag_name]
        feats = np.array([p[1] for p in patches])
        ins_names = [p[0] for p in patches]
        label = 1 if "tumor" in bag_name.lower() else 0
        return torch.FloatTensor(feats), torch.LongTensor([label]), ins_names, bag_name
