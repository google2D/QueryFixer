!pip install torch nltk

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize

# Download NLTK data files for tokenization
nltk.download('punkt')

# Load the training data
df = pd.read_csv("sample_data/train.tsv", sep="\t", names=["query", "score"])

# Tokenization using NLTK
def tokenize(text):
    return word_tokenize(text.lower())  # Lowercase and tokenize

# Build a vocabulary
counter = Counter()
for query in df['query']:
    counter.update(tokenize(query))

# Create a vocabulary object
vocab = {token: idx for idx, (token, _) in enumerate(counter.items())}

# Tokenize the queries and convert to tensor indices
def tokenize_and_encode(queries):
    return [torch.tensor([vocab[token] for token in tokenize(query)]) for query in queries]

encoded_queries = tokenize_and_encode(df['query'].tolist())

# Pad sequences and create input tensor
max_length = 128
input_tensor = torch.nn.utils.rnn.pad_sequence(encoded_queries, batch_first=True, padding_value=0)

# Create a TensorDataset
dataset = TensorDataset(input_tensor, torch.tensor(df['score'].tolist(), dtype=torch.long))  # Ensure labels are Long tensor

# Create DataLoader with a smaller batch size for testing
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        out = self.fc(hidden[-1])
        return out

# Instantiate the model
vocab_size = len(vocab)
embed_dim = 100
hidden_dim = 64
output_dim = 2  # Binary classification (well-formed or not)

model = LSTMModel(vocab_size, embed_dim, hidden_dim, output_dim)

# Move the model to the appropriate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Set up the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Set number of epochs (use a smaller number for testing)
epochs = 4

# Training loop
for epoch in range(epochs):
    for batch in dataloader:
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs} complete. Loss: {loss.item()}")
