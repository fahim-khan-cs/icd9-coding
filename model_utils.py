import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader

class ArrayDataset(Dataset):
    def __init__(self, X_data, y_data):
        """
        X_data: A list of arrays, each 700x100.
        y_data: A list of arrays, each 1x6918 (one-hot encoded labels).
        """
        self.X_data = torch.FloatTensor(X_data)
        # assert self.X_data.shape[1] == 100
        self.y_data = torch.FloatTensor(y_data)
        # assert self.y_data.shape[1] == 6918

    def __len__(self):
        return self.X_data.shape[0]

    def __getitem__(self, idx):
        # Extract the document and its corresponding label
        X = self.X_data[idx, :, :]
        y = self.y_data[idx]

        # No need to reshape y since it's already in the correct format for one-hot encoded vectors
        return X, y
    

class D2VDataset(Dataset):
    def __init__(self, X_data, y_data):
        """
        X_data: A list of arrays, each 700x100.
        y_data: A list of arrays, each 1x6918 (one-hot encoded labels).
        """
        self.X_data = torch.FloatTensor(X_data)
        # assert self.X_data.shape[1] == 100
        self.y_data = torch.FloatTensor(y_data)
        # assert self.y_data.shape[1] == 6918

    def __len__(self):
        return self.X_data.shape[0]

    def __getitem__(self, idx):
        # Extract the document and its corresponding label
        X = self.X_data[idx]
        y = self.y_data[idx]

        # No need to reshape y since it's already in the correct format for one-hot encoded vectors
        return X, y

def score_model(trues, outputs, thresh=0.2):
  # Score CNN Part model function
  # Default threshold = 0.2
  with torch.no_grad():
    outputs = np.array(outputs).ravel()
    trues = np.array(trues).ravel()
    preds = np.array([outputs>=thresh], dtype=np.float32).ravel()

    # correct_predictions = (trues == preds).sum()

    true_positives = (
      trues[trues==1] == preds[trues==1]).sum()
    false_positives = (
      preds[preds==1] != trues[preds==1]).sum()
    false_negatives = (
      preds[preds==0] != trues[preds==0]).sum()
    true_negatives = (
      trues[trues==0] == preds[trues==0]).sum()

    precision = true_positives/(true_positives + false_positives+ 1e-10)
    recall = true_positives/(true_positives + false_negatives+ 1e-10)
    f1 = 2*precision*recall/(precision+recall)

    return precision, recall, f1

def evaluate(model, data_loader, criterion):

  model.eval()  # Setting the model to evaluation mode

  all_true_labels = []
  all_outputs = []

  losses = []
  metrics_ls = []

  with torch.no_grad():  # No need to track gradients during evaluation
    for inputs, labels in data_loader:
      outputs = model(inputs)

      all_outputs.extend(outputs.cpu().numpy())
      all_true_labels.extend(labels.cpu().numpy())

      loss = criterion(outputs, labels.float())
      losses.append(loss.item())

      metrics = score_model(labels.ravel(), outputs.ravel())
      metrics_ls.append(metrics)

  prec, rec, f1 = zip(*metrics_ls)
  return np.mean(losses), np.mean(prec), np.mean(rec), np.mean(f1)

def train_model(model, data_loader, criterion, optimizer):

  model.train()
  losses = []
  metrics_ls = []

  for inputs,labels in data_loader:

    # Zero the parameter gradients
    optimizer.zero_grad()

    # Forward + backward + optimize
    outputs = model(inputs)

    loss = criterion(outputs, labels.float())

    loss = criterion(outputs, labels.float())
    losses.append(loss.item())

    metrics = score_model(labels, outputs)
    metrics_ls.append(metrics)

    loss.backward()
    optimizer.step()

  prec, rec, f1 = zip(*metrics_ls)

  # for i in losses, prec, recs, f1s:
  # print(np.array(i).shape)

  return np.mean(losses), np.mean(prec), np.mean(rec), np.mean(f1)

