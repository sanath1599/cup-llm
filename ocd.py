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
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def load_data(file_path):
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
    return data

def get_embeddings(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze()  

def extract_features(data):
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
    def __init__(self):
        super().__init__()
        self.attention = nn.Linear(3072, 1)
        self.fc1 = nn.Linear(3072, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        weights = self.sigmoid(self.attention(x))
        weighted_features = weights * x
        x = self.relu(self.fc1(weighted_features))
        x = self.sigmoid(self.fc2(x))
        return x

def train_model(data_loader, model, criterion, optimizer, epochs=10):
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
