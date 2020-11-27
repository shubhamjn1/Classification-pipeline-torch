import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from tqdm import tqdm

class pyClassifier():
    """
    """
    def __init__(self, X_train, X_test, y_train, y_test):
        # super.__init__()
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        # set parameters for NN
        self.NUM_FEATURES = (self.X_train.shape[1])
        self.NUM_CLASSES = len(np.unique(self.y_train))
    
    
    def processData(self, batch_size = 16):
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

    
    def fitNN(self, epoch = 10, lr = 0.01):
        """
        docstring
        """
        # dataprep
        train_loader, test_loader = self.processData()
        
        # setup
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print (f"Using {device}")
        
        # model call and loss
        model = classifierNN(num_features=self.NUM_FEATURES, num_class=self.NUM_CLASSES)
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr = lr)

        print (model)

        # saving acc and losses per epoch
        accuracy_stats = {'train':[], 'val':[]}
        loss_stats = {'train':[], 'val':[]}

        # training starts
        print ("Begin Training")
        for e in tqdm(range(1, epoch+1)):
            # TRAINING
            train_epoch_loss = 0
            train_epoch_acc = 0

            model.train()
            for X_train_batch, y_train_batch in train_loader:
                X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
                optimizer.zero_grad()

                y_train_pred = model(X_train_batch)
                train_loss = criterion(y_train_pred, y_train_batch)
                train_acc = self.multi_acc(y_train_pred, y_train_batch)
                
                train_loss.backward()
                optimizer.step()

                train_epoch_loss += train_loss.item()
                train_epoch_acc += train_acc.item()

            loss_stats['train'].append(train_epoch_loss/len(train_loader))
            accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
            
            print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} \
                    Train Acc: {train_epoch_acc/len(train_loader):.3f}')

        return model, accuracy_stats, loss_stats


    def multi_acc(self, y_pred, y_test):
        y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
        _, y_pred_labels = torch.max(y_pred_softmax, dim = 1)

        correct_predictions = (y_pred_labels == y_test).float()
        acc = correct_predictions.sum() / len(correct_predictions)

        acc = torch.round(acc)*100
        return acc
        
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
        # super.__init__()
        super(classifierNN, self).__init__()

        self.layer1 = nn.Linear(num_features, 512)
        self.layer2 = nn.Linear(512, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer_out = nn.Linear(64, num_class)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = self.layer1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)

        x = self.layer2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_out(x)

        return x



if __name__ == "__main__":
    data = load_wine(as_frame=True)
    target_values = data['target']
    data = data['data']

    # split into train test
    X_train, X_test, y_train, y_test = train_test_split(data, target_values, test_size=0.1, \
        stratify=target_values, random_state=2020)
    
    # normalize the data
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)

    print (f"Train size:{X_train.shape}, Test size:{X_test.shape}")
    classifier = pyClassifier(X_train, X_test, y_train, y_test)
    classifier.fitNN()
