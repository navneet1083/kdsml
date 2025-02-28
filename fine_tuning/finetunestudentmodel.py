import torch.nn as nn


class FineTuneStudentModel(nn.Module):
    def __init__(self, student_model, num_labels=2):
        """
        Wrap the pre-trained student model by extracting its embedding and transformer layers,
        and adding a classification head.
        """
        super(FineTuneStudentModel, self).__init__()
        self.student_model = student_model
        # Use the pre-trained student's embedding to get the hidden size.
        self.hidden_size = self.student_model.embedding.embedding_dim
        self.classifier = nn.Linear(self.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None):
        # Pass input_ids through embedding
        x = self.student_model.embedding(input_ids)  # (batch, seq, hidden_size)
        for layer in self.student_model.layers:
            if attention_mask is not None:
                x = layer(x.transpose(0, 1), src_key_padding_mask=~attention_mask.bool()).transpose(0, 1)
            else:
                x = layer(x.transpose(0, 1)).transpose(0, 1)
        # Use the [CLS] token (first token) for classification.
        cls_token = x[:, 0, :]
        logits = self.classifier(cls_token)
        return logits