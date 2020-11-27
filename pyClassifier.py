import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from torch.optim as optim

class pyClassifier():
    """
    """
    def __init__(self, X_train, X_test, y_train, y_test):
        super.__init__()
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        # set parameters for NN
        self.NUM_FEATURES = len(self.X_train.columns)
        self.NUM_CLASSES = len(np.unique(self.y_train))
    
    @staticmethod
    def processData(self):
        """
        docstring
        """

        # prep data into tensors
        train_dataset = ClassifierDataset(torch.from_numpy(self.X_train).float(), \
            torch.from_numpy(self.y_train).long())

        test_dataset = ClassifierDataset(torch.from_numpy(self.X_test).float(), \
            torch.from_numpy(self.y_test).long())

        # dataloaders
        train_loader = DataLoader(dataset = train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)
        
        test_loader = DataLoader(dataset = test_dataset, batch_size=1)

        return train_loader, test_loader

    
    def fitNN(self, epoch = 10, batch_size = 16, lr = 0.01):
        """
        docstring
        """
        # dataprep
        train_loader, test_loader = processData()
        
        # setup
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print (f"Using {device}")
        
        # model call and loss
        model = classifierNN(num_features=self.NUM_FEATURES, num_class=self.NUM_CLASSES)
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr = lr)

        print (model)
        

        
class ClassifierDataset(Dataset):
    """To prepare item level dataset for NN

    Args:
        Dataset ([type]): [description]
    """
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)

class classifierNN(nn.Module):
    """DNN architecture

    Args:
        nn ([type]): [description]
    """
    def __init__(self, num_features, num_class):
        super.__init__()

        self.layer1 = nn.Linear(num_features, 512)
        self.layer2 = nn.Linear(512, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer_out = nn.Linear(64, num_class)

        self.relu = nn.ReLU
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)

        x = self.layer2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_out(x)

        return x



if __name__ == "__main__":
    data = load_wine(as_frame=True)
    target_values = data['target']
    data = data['data']

    # split into train test
    X_train, X_test, y_train, y_test = train_test_split(data, target_values, test_size=0.25, \
        stratify=target_values, random_state=2020)
    
    # normalize the data
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)

