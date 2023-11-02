import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# Define a custom AttentionModel class that inherits from nn.Module
class AttentionModel(nn.Module):

    def __init__(self, vec_len, seq_len, n_classes):
        super(AttentionModel, self).__init__()

        # Initialize the class attributes
        self.vec_len = vec_len
        self.seq_len = seq_len

        # Initialize attention weights with a small random value
        self.attn_weights = torch.cat([torch.tensor([[0.]]),
                                       torch.randn(vec_len, 1) /
                                       torch.sqrt(torch.tensor(vec_len))])

        # Make the attention weights a trainable parameter
        self.attn_weights.requires_grad = True
        self.attn_weights = nn.Parameter(self.attn_weights)

        # Define activation and softmax functions
        self.activation = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

        # Linear layer for the final output
        self.linear = nn.Linear(vec_len + 1, n_classes)

    def forward(self, input_data):
        # Calculate the attention weights and apply them to the input data
        hidden = torch.matmul(input_data, self.attn_weights)
        hidden = self.activation(hidden)
        attn = self.softmax(hidden)
        attn = attn.repeat(1, 1, self.vec_len + 1).reshape(attn.shape[0],
                                                           self.seq_len,
                                                           self.vec_len + 1)
        attn_output = input_data * attn
        attn_output = torch.sum(attn_output, axis=1)
        output = self.linear(attn_output)
        return output

# Define a function for training the model
def train(train_loader, valid_loader, model, criterion, optimizer, device,
          num_epochs, model_path):
    """
    Function to train the model
    :param train_loader: Data loader for train dataset
    :param valid_loader: Data loader for validation dataset
    :param model: Model object
    :param criterion: Loss function
    :param optimizer: Optimizer
    :param device: CUDA or CPU
    :param num_epochs: Number of epochs
    :param model_path: Path to save the model
    """
    best_loss = 1e8
    for i in range(num_epochs):
        print(f"Epoch {i+1} of {num_epochs}")
        valid_loss, train_loss = [], []
        model.train()

        # Train loop
        for batch_labels, batch_data in tqdm(train_loader):
            # Move data to GPU if available
            batch_labels = batch_labels.to(device)
            batch_data = batch_data.to(device)

            # Forward pass
            batch_output = model(batch_data)
            batch_output = torch.squeeze(batch_output)

            # Calculate loss
            loss = criterion(batch_output, batch_labels)
            train_loss.append(loss.item())
            optimizer.zero_grad()

            # Backward pass
            loss.backward()

            # Gradient update step
            optimizer.step()

        model.eval()

        # Validation loop
        for batch_labels, batch_data in tqdm(valid_loader):
            # Move data to GPU if available
            batch_labels = batch_labels.to(device)
            batch_data = batch_data.to(device)

            # Forward pass
            batch_output = model(batch_data)
            batch_output = torch.squeeze(batch_output)

            # Calculate loss
            loss = criterion(batch_output, batch_labels)
            valid_loss.append(loss.item())

        t_loss = np.mean(train_loss)
        v_loss = np.mean(valid_loss)
        print(f"Train Loss: {t_loss}, Validation Loss: {v_loss}")

        if v_loss < best_loss:
            best_loss = v_loss

            # Save model if validation loss improves
            torch.save(model.state_dict(), model_path)

        print(f"Best Validation Loss: {best_loss}")

# Define a function for testing the model
def test(test_loader, model, criterion, device):
    """
    Function to test the model
    :param test_loader: Data loader for test dataset
    :param model: Model object
    :param criterion: Loss function
    :param device: CUDA or CPU
    """
    model.eval()
    test_loss = []
    test_accu = []
    for batch_labels, batch_data in tqdm(test_loader):
        # Move data to device
        batch_labels = batch_labels.to(device)
        batch_data = batch_data.to(device)

        # Forward pass
        batch_output = model(batch_data)
        batch_output = torch.squeeze(batch_output)

        # Calculate loss
        loss = criterion(batch_output, batch_labels)
        test_loss.append(loss.item())
        batch_preds = torch.argmax(batch_output, axis=1)

        # Move predictions to CPU
        if torch.cuda.is_available():
            batch_labels = batch_labels.cpu()
            batch_preds = batch_preds.cpu()

        # Compute accuracy
        test_accu.append(accuracy_score(batch_labels.detach().numpy(),
                                        batch_preds.detach().numpy()))

    test_loss = np.mean(test_loss)
    test_accu = np.mean(test_accu)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accu}")
