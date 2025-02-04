# models/student_model.py
import torch
import torch.nn as nn

class StudentModel(nn.Module):
    def __init__(self, hidden_size=384, num_layers=6):
        super(StudentModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=30522, embedding_dim=hidden_size)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8) for _ in range(num_layers)])
        self.fc = nn.Linear(hidden_size, 30522)  # Output vocabulary size

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.fc(x)