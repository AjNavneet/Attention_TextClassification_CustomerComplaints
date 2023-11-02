import re
import torch
import config
import argparse
from Source.utils import load_file
from Source.model import AttentionModel
from nltk.tokenize import word_tokenize

# Define the main function to perform text classification
def main(args_):
    # Process the input test complaint
    input_text = args_.test_complaint
    input_text = input_text.lower()  # Convert text to lowercase
    input_text = re.sub(r"[^\w\d'\s]+", " ", input_text)  # Remove punctuation except apostrophes
    input_text = re.sub("\d+", "", input_text)  # Remove digits
    input_text = re.sub(r'[x]{2,}', "", input_text)  # Remove consecutive 'x' characters
    input_text = re.sub(' +', ' ', input_text)  # Remove extra spaces
    tokens = word_tokenize(input_text)  # Tokenize the cleaned text
    
    # Add padding if the length of tokens is less than 20
    tokens = ['<pad>'] * (20 - len(tokens)) + tokens
    
    # Load label encoder and determine the number of classes
    label_encoder = load_file(config.label_encoder_path)
    num_classes = len(label_encoder.classes_)
    
    # Determine the device (CPU or GPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load the pre-trained model
    model = AttentionModel(config.vec_len, config.seq_len, num_classes, device)
    model_path = config.model_path
    model.load_state_dict(torch.load(model_path))
    
    # Move the model to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Load vocabulary and word embeddings
    vocabulary = load_file(config.vocabulary_path)
    embeddings = load_file(config.embeddings_path)
    
    # Tokenize the input text and map tokens to their embeddings
    idx_token = []
    for token in tokens:
        if token in vocabulary:
            idx_token.append(vocabulary.index(token))
        else:
            idx_token.append(vocabulary.index('<unk>'))  # Handle unknown tokens
    
    # Select a maximum of 20 tokens (config.seq_len)
    token_emb = embeddings[idx_token, :]
    token_emb = token_emb[:config.seq_len, :]
    
    # Convert token embeddings to a torch tensor
    inp = torch.from_numpy(token_emb)
    
    # Add a '1' vector to account for the bias term
    inp = torch.cat((torch.ones(inp.shape[0], 1), inp), dim=1)
    
    # Move the tensor to the GPU if available
    inp = inp.to(device)
    
    # Create a batch of 1 data point
    inp = torch.unsqueeze(inp, 0)
    
    # Forward pass through the model
    out = torch.squeeze(model(inp))
    
    # Find the predicted class based on the highest predicted value
    prediction = label_encoder.classes_[torch.argmax(out)]
    print(f"Predicted Class: {prediction}")

if __name__ == "__main__":
    # Parse the test complaint from the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_complaint", type=str, help="Test complaint")
    args = parser.parse_args()
    main(args)
