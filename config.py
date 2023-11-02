# Learning rate for training
lr = 0.001

# Length of word embeddings
vec_len = 50

# Maximum sequence length for input data
seq_len = 20

# Number of training epochs
num_epochs = 50

# Column name containing labels (e.g., product categories)
label_col = "Product"

# Path to the file containing tokenized text data
tokens_path = "Output/tokens.pkl"

# Path to the file containing label data
labels_path = "Output/labels.pkl"

# Path to the data file (CSV) with text and labels
data_path = "Input/complaints.csv"

# Path to save the trained model
model_path = "Output/attention.pth"

# Path to save the vocabulary (word-to-index mapping)
vocabulary_path = "Output/vocabulary.pkl"

# Path to save word embeddings
embeddings_path = "Output/embeddings.pkl"

# Path to the GloVe word vectors file
glove_vector_path = "Input/glove.6B.50d.txt"

# Column name containing text data (complaint narratives)
text_col_name = "Consumer complaint narrative"

# Path to save the label encoder (used for encoding and decoding labels)
label_encoder_path = "Output/label_encoder.pkl"

# Mapping of product names to category labels
product_map = {
    'Vehicle loan or lease': 'vehicle_loan',
    'Credit reporting, credit repair services, or other personal consumer reports': 'credit_report',
    'Credit card or prepaid card': 'card',
    'Money transfer, virtual currency, or money service': 'money_transfer',
    'virtual currency': 'money_transfer',
    'Mortgage': 'mortgage',
    'Payday loan, title loan, or personal loan': 'loan',
    'Debt collection': 'debt_collection',
    'Checking or savings account': 'savings_account',
    'Credit card': 'card',
    'Bank account or service': 'savings_account',
    'Credit reporting': 'credit_report',
    'Prepaid card': 'card',
    'Payday loan': 'loan',
    'Other financial service': 'others',
    'Virtual currency': 'money_transfer',
    'Student loan': 'loan',
    'Consumer Loan': 'loan',
    'Money transfers': 'money_transfer'
}
