import json
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split

code_tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
code_model = AutoModel.from_pretrained("microsoft/graphcodebert-base")
comment_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
comment_model = AutoModel.from_pretrained("bert-base-uncased")

class CodeCommentDataset(Dataset):
    """
    A dataset class for code comment data.

    Args:
        features (list): List of code comment features.
        labels (list): List of corresponding labels.

    Attributes:
        features (list): List of code comment features.
        labels (list): List of corresponding labels.
    """

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.features)
    
    def __getitem__(self, idx):
        """
        Returns a specific item from the dataset.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            tuple: A tuple containing the feature and label at the specified index.
        """
        return self.features[idx], self.labels[idx]

def load_data(file_path):
    """
    Load data from a JSON file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        list: A list of dictionaries representing the loaded data.
    """
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
    return data

def get_embeddings(model, tokenizer, text):
    """
    Get the embeddings for a given text using a pre-trained model and tokenizer.

    Args:
        model (object): The pre-trained model used for generating embeddings.
        tokenizer (object): The tokenizer used for tokenizing the text.
        text (str): The input text for which embeddings need to be generated.

    Returns:
        torch.Tensor: The embeddings for the input text.
    """
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze()

def extract_features(data):
    """
    Extracts features from the given data.

    Args:
        data (list): A list of samples, where each sample is a dictionary containing the following keys:
            - 'src_method': The source code of the method.
            - 'dst_method': The destination code of the method.
            - 'src_desc_tokens': The tokens of the source code description.
            - 'dst_desc_tokens': The tokens of the destination code description.
            - 'label': The label associated with the sample.

    Returns:
        torch.Tensor: A tensor containing the combined features of all samples.
        torch.Tensor: A tensor containing the labels of all samples.
    """
    features = []
    labels = []
    for sample in data:
        src_code_emb = get_embeddings(code_model, code_tokenizer, sample['src_method'])
        dst_code_emb = get_embeddings(code_model, code_tokenizer, sample['dst_method'])
        src_comment_emb = get_embeddings(comment_model, comment_tokenizer, ' '.join(sample['src_desc_tokens']))
        dst_comment_emb = get_embeddings(comment_model, comment_tokenizer, ' '.join(sample['dst_desc_tokens']))
        combined_features = torch.cat((src_code_emb, dst_code_emb, src_comment_emb, dst_comment_emb))
        features.append(combined_features)
        labels.append(torch.tensor(sample['label']))
        
    return torch.stack(features), torch.stack(labels)

class SimpleAttentionClassifier(nn.Module):
    """
    A simple attention-based classifier module.

    This module applies attention mechanism to the input features and performs classification.

    Args:
        None

    Attributes:
        attention (nn.Linear): Linear layer for attention calculation.
        fc1 (nn.Linear): Linear layer for the first fully connected layer.
        relu (nn.ReLU): ReLU activation function.
        fc2 (nn.Linear): Linear layer for the second fully connected layer.
        sigmoid (nn.Sigmoid): Sigmoid activation function.

    Methods:
        forward(x): Performs forward pass through the network.

    """

    def __init__(self):
        super().__init__()
        self.attention = nn.Linear(3072, 1)
        self.fc1 = nn.Linear(3072, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        Performs forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).

        """
        weights = self.sigmoid(self.attention(x))
        weighted_features = weights * x
        x = self.relu(self.fc1(weighted_features))
        x = self.sigmoid(self.fc2(x))
        return x

def train_model(data_loader, model, criterion, optimizer, epochs=10):
    """
    Trains the given model using the provided data loader, criterion, and optimizer.

    Args:
        data_loader (torch.utils.data.DataLoader): The data loader object that provides the training data.
        model (torch.nn.Module): The model to be trained.
        criterion (torch.nn.Module): The loss function used to compute the training loss.
        optimizer (torch.optim.Optimizer): The optimizer used to update the model's parameters.
        epochs (int, optional): The number of training epochs. Defaults to 10.

    Returns:
        None
    """
    model.train()
    for epoch in range(epochs):
        for features, labels in data_loader:
            optimizer.zero_grad()
            outputs = model(features.float())
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Training Loss: {loss.item()}")

def evaluate_model(data_loader, model):
    """
    Evaluate the performance of a model on a given data loader.

    Args:
        data_loader (torch.utils.data.DataLoader): The data loader containing the evaluation data.
        model: The model to be evaluated.

    Returns:
        None
    """
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for features, labels in data_loader:
            outputs = model(features.float()).squeeze()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {correct / total}")

def save_predictions(data_loader, model, file_path):
    """
    Save the predictions made by the model on the given data_loader to a file.

    Args:
        data_loader (torch.utils.data.DataLoader): The data loader containing the input data.
        model (torch.nn.Module): The trained model used for making predictions.
        file_path (str): The path to the file where the predictions will be saved.

    Returns:
        None
    """
    model.eval()
    results = []
    with torch.no_grad():
        for features, labels in data_loader:
            outputs = model(features.float()).squeeze()
            predicted = (outputs > 0.5).float()
            results.extend(predicted.tolist())
    
    with open(file_path, 'w') as f:
        for result in results:
            f.write(f"{result}\n")


data = load_data("valid.jsonl")
features, labels = extract_features(data)
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)


train_dataset = CodeCommentDataset(features_train, labels_train)
test_dataset = CodeCommentDataset(features_test, labels_test)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)


model = SimpleAttentionClassifier()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


train_model(train_loader, model, criterion, optimizer)
evaluate_model(test_loader, model)

save_predictions(test_loader, model, "predictions.txt")
