import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import pandas as pd
import numpy as np

# train the Model
def train_loop(data_loader, model, loss_fn, optimizer, loss_estimate, batch_no, epoch, epoch_no):
    """
    Training loop for the machine learning model.

    Parameters:
    - data_loader (torch.utils.data.DataLoader): DataLoader containing training data.
    - model (torch.nn.Module): The machine learning model to be trained.
    - loss_fn: The loss function used for training.
    - optimizer: The optimization algorithm for updating model parameters.
    - loss_estimate (list): List to store loss values for visualization.
    - batch_no (list): List to store batch numbers for visualization.
    - epoch (int): The current epoch number.
    - epoch_no (list): List to store epoch numbers for visualization.
    """
    size = len(data_loader.dataset)
    model.train()

    # Iterate through batches in the data loader
    for batch, (X, y) in enumerate(data_loader):
        # Forward pass
        pred = model(X)

        # Compute the loss
        loss = loss_fn(pred, y)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Logging and visualization
        if batch % 1000 == 0:
            loss_value, current_batch = loss.item(), (batch + 1) * len(X)

            # Append values for visualization
            loss_estimate.append(loss_value)
            batch_no.append(current_batch)
            epoch_no.append(epoch)

            # Print progress
            print(f"loss: {loss_value:>7f}  [{current_batch:>5d}/{size:>5d}]")

def test_loop(data_loader, model, loss_fn):
    """
    Testing loop for evaluating the performance of a machine learning model on a test dataset.

    Parameters:
    - data_loader (torch.utils.data.DataLoader): DataLoader containing test data.
    - model (torch.nn.Module): The trained machine learning model.
    - loss_fn: The loss function used for evaluation.
    """
    size = len(data_loader.dataset)
    model.eval()
    num_batches = len(data_loader)
    test_loss, correct = 0, 0

    # Disable gradient computation during evaluation
    with torch.no_grad():
        # Iterate through batches in the data loader
        for (X, y) in data_loader:
            # Forward pass
            pred = model(X)

            # Compute test loss
            test_loss += loss_fn(pred, y).item()

            # Calculate the number of correct predictions
            correct += (pred.argmax(dim = 1) == y.argmax(dim=1)).type(torch.float).sum().item()

    # Calculate average test loss and accuracy
    test_loss /= num_batches
    accuracy = correct / size

    # Print test results
    print(f"Test Error: \n Accuracy: {(100 * accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")

class CustomDatasetTrain(Dataset):
    """
    Custom PyTorch dataset for training data.

    Parameters:
    - X_train: Features of the training dataset.
    - y_train: Labels of the training dataset.
    """
    def __init__(self, X_train, y_train):
        self.features = X_train
        self.label = y_train

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.label)

    def __getitem__(self, idx):
        """
        Returns a sample from the dataset given an index.

        Parameters:
        - idx (int): Index of the sample.

        Returns:
        - features (tensor): Features of the sample.
        - label (tensor): Label of the sample.
        """
        features = self.features[idx]
        label = self.label[idx]
        return features, label

class extract_tensor(nn.Module):
    def forward(self,x):
        # Output shape (batch, features, hidden)
        tensor, _ = x
        # Reshape shape (batch, hidden)
        return tensor[:, -1, :]

class NeuralNetwork(nn.Module):
    """
    Definition of a neural network model with an LSTM stack for a specific task.

    Architecture:
    - Embedding layer with input dimension 64, output dimension 32, max_norm regularization, and L2 normalization.
    - Bidirectional LSTM layer with input size 32, hidden size 64, 1 layer, batch-first, and 20% dropout.
    - Custom function extract_tensor() (please provide the implementation).
    - Linear layer with input size 128 and output size 26.

    Parameters:
    - None

    Input:
    - x (torch.Tensor): Input tensor to be processed by the neural network.

    Output:
    - logits (torch.Tensor): Output logits produced by the neural network.
    """
    def __init__(self):
        super().__init__()
        self.LSTM_stack = nn.Sequential(
            nn.Embedding(36, 6, max_norm=1, norm_type=2),
            nn.LSTM(input_size=6, hidden_size=36, num_layers=1, batch_first=True, dropout=0.2, bidirectional=True),
            extract_tensor(),  # Please provide the implementation of extract_tensor()
            nn.Linear(72, 26)
        )

    def forward(self, x):
        logits = self.LSTM_stack(x)
        return logits

def create_dataloader(input_tensor, target_tensor):
    all_features_data = CustomDatasetTrain(input_tensor, target_tensor)
    all_features_dataloader = DataLoader(all_features_data, batch_size=128, shuffle=True)
    return all_features_dataloader

def save_model(model):
    torch.save(model.state_dict(), "lstm_ngram_2.pt")

def train_model(input_tensor, target_tensor):
    """
    Trains a neural network model using the specified input and target tensors.

    Parameters:
    - input_tensor (torch.Tensor): Input data tensor.
    - target_tensor (torch.Tensor): Target data tensor.
    """
    # Create a DataLoader for the training data
    all_features_dataloader = create_dataloader(input_tensor, target_tensor)

    # Initialize the neural network model
    model = NeuralNetwork()

    # Define the loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # Lists for storing loss, batch, and epoch values for visualization
    loss_estimate = []
    batch_no = []
    epoch_no = []

    # Number of training epochs
    epochs = 5

    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")

        # Train the model
        train_loop(all_features_dataloader, model, loss_fn, optimizer, loss_estimate, batch_no, epoch, epoch_no)

        # Evaluate on the test set
        test_loop(all_features_dataloader, model, loss_fn)

    print("Training complete!")

    # Save the trained model
    save_model(model)
