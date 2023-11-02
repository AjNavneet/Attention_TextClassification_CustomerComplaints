import torch

class TextDataset(torch.utils.data.Dataset):

    def __init__(self, tokens, embeddings, labels):
        """
        Initialize the TextDataset.

        Args:
            tokens (list): List of word tokens as integer indices.
            embeddings (numpy array): Pre-trained word embeddings (e.g., from GloVe).
            labels (list): List of labels corresponding to the tokens.
        """
        self.tokens = tokens
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        """
        Get the total number of data points in the dataset.

        Returns:
            int: The number of data points.
        """
        return len(self.tokens)

    def __getitem__(self, idx):
        """
        Get a specific data point from the dataset.

        Args:
            idx (int): Index of the data point to retrieve.

        Returns:
            tuple: A tuple containing label and input data.
        """
        # Retrieve the word embeddings for the tokens at the given index
        emb = torch.tensor(self.embeddings[self.tokens[idx], :])
        
        # Construct the input data by adding a bias term (a vector of ones) to the embeddings
        input_ = torch.cat((torch.ones(emb.shape[0], 1), emb), dim=1)
        
        # Return a tuple with label and input data
        return torch.tensor(self.labels[idx]), input_
