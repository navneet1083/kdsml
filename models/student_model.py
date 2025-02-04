import torch
import torch.nn as nn

class StudentModel(nn.Module):
    def __init__(self, hidden_size=384, num_layers=6, vocab_size=30522):
        super(StudentModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size)
        self.layers = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8) for _ in range(num_layers)])
        self.fc = nn.Linear(hidden_size, vocab_size)  # Output vocabulary size

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)

        # Apply transformer layers
        for layer in self.layers:
            if attention_mask is not None:
                x = layer(x.transpose(0, 1), src_key_padding_mask=~attention_mask.bool()).transpose(0, 1)
            else:
                x = layer(x.transpose(0, 1)).transpose(0, 1)

        return self.fc(x)  # Shape: (batch_size, sequence_length, vocab_size=30522)