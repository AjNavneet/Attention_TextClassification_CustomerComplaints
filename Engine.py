import torch
import config
from Source.data import TextDataset
from Source.utils import load_file, save_file
from sklearn.model_selection import train_test_split
from Source.model import AttentionModel, train, test

# Define a main function to encapsulate the script logic
def main():
    # Load tokenized text data, labels, word embeddings, label encoder, and other configuration from config.py
    print("Loading the files...")
    tokens = load_file(config.tokens_path)
    labels = load_file(config.labels_path)
    embeddings = load_file(config.embeddings_path)
    label_encoder = load_file(config.label_encoder_path)
    num_classes = len(label_encoder.classes_)

    # Split the data into training, validation, and test sets
    print("Splitting data into train, valid and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(tokens, labels, test_size=0.2)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25)

    # Create PyTorch datasets for training, validation, and testing
    print("Creating PyTorch datasets...")
    train_dataset = TextDataset(X_train, embeddings, y_train)
    valid_dataset = TextDataset(X_valid, embeddings, y_valid)
    test_dataset = TextDataset(X_test, embeddings, y_test)

    # Create data loaders for training, validation, and testing
    print("Creating data loaders...")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=16)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16)

    # Determine the device (CPU or GPU) for model training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create an instance of the AttentionModel with specified dimensions and classes
    print("Creating model object...")
    model = AttentionModel(config.vec_len, config.seq_len, num_classes, device)
    model_path = config.model_path

    # Move the model to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()

    # Define the loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # Train the model
    print("Training the model...")
    train(train_loader, valid_loader, model, criterion, optimizer, device, config.num_epochs, model_path)

    # Test the model
    print("Testing the model...")
    test(test_loader, model, criterion, device)

# Entry point of the script
if __name__ == "__main__":
    main()
