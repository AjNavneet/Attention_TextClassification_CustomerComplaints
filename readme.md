# Attention Mechanism based Text Classification on Customer Complaints Data

### Business Context

The attention mechanism is a powerful tool that dynamically highlights relevant features of input data. It assigns more weight to the most relevant words in a sentence, improving the quality of predictions. This project delves into the application of the attention mechanism for text classification, evaluating its effectiveness in solving our problem.

---

### Aim

To perform multiclass text classification using the attention mechanism on a dataset of customer complaints about consumer financial products.

---

### Data Description

The dataset consists of more than two million customer complaints about consumer financial products. It includes columns for the actual text of the complaint and the product category associated with each complaint. Pre-trained word vectors from the GloVe dataset (glove.6B) are used to enhance text representation.

---

### Tech Stack

- Language: `Python`
- Libraries: `pandas`, `torch`, `nltk`, `numpy`, `pickle`, `re`, `tqdm`, `sklearn`

---

## Approach

1. Installing the necessary packages via `pip`
2. Importing the required libraries
3. Defining configuration file paths
4. Processing GloVe embeddings:
   - Reading the text file
   - Converting embeddings to float arrays
   - Adding embeddings for padding and unknown items
   - Saving embeddings and vocabulary
5. Processing Text data:
   - Reading the CSV file and dropping null values
   - Replacing duplicate labels
   - Encoding the label column and saving the encoder and encoded labels
6. Data Preprocessing:
   - Conversion to lowercase
   - Punctuation removal
   - Digits removal
   - Removing consecutive instances of 'x'
   - Removing additional spaces
   - Tokenizing the text
   - Saving the tokens
7. Model:
   - Creating the attention model
   - Defining a function for the PyTorch dataset
   - Functions for training and testing the model
8. Training:
   - Loading the necessary files
   - Splitting data into train, test, and validation sets
   - Creating PyTorch datasets
   - Creating data loaders
   - Creating the model object
   - Moving the model to GPU if available
   - Defining the loss function and optimizer
   - Training the model
   - Testing the model
9. Making predictions on new text data

---

## Modular Code Overview

1. **Input**: Contains the data required for analysis, including:
   - `complaints.csv`
   - `glove.6B.50d.txt` (download from [here](https://nlp.stanford.edu/projects/glove/))

2. **Source**: Contains modularized code for various project steps, including:
   - `model.py`
   - `data.py`
   - `utils.py`

   These Python files contain helpful functions used in `Engine.py`.

3. **Output**: Contains files required for model training, including:
   - `attention.pth`
   - `embeddings.pkl`
   - `label_encoder.pkl`
   - `labels.pkl`
   - `vocabulary.pkl`
   - `tokens.pkl`

4. **config.py**: Contains project configurations.

5. **Engine.py**: The main file to run the entire project, which trains the model and saves it in the output folder.

---

## Key Concepts Explored

1. Understanding the business problem
2. Introduction to the attention mechanism
3. Understanding how the attention mechanism works
4. Working with pre-trained word vectors
5. Steps to process GloVe embeddings
6. Data preparation for the models
7. Handling spaces and digits
8. Punctuation removal
9. Architecting the attention model
10. Creating a data loader for the attention model
11. Building the attention model
12. Training the attention model using GPU or CPU
13. Making predictions on new text data
14. Understanding how to use attention with RNN

---

