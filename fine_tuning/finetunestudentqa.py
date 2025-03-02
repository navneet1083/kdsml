import torch.nn as nn


class FineTuneStudentQA(nn.Module):
    def __init__(self, student_model):
        """
        This wrapper uses the pre-trained student model's embedding and transformer layers
        and adds a QA head (a linear layer) to produce start and end logits.
        """
        super(FineTuneStudentQA, self).__init__()
        self.student_model = student_model
        self.hidden_size = self.student_model.embedding.embedding_dim
        self.qa_outputs = nn.Linear(self.hidden_size, 2)

    def forward(self, input_ids, attention_mask=None):
        x = self.student_model.embedding(input_ids)  # (batch, seq, hidden_size)
        for layer in self.student_model.layers:
            if attention_mask is not None:
                x = layer(x.transpose(0, 1), src_key_padding_mask=~attention_mask.bool()).transpose(0, 1)
            else:
                x = layer(x.transpose(0, 1)).transpose(0, 1)
        logits = self.qa_outputs(x)  # (batch, seq, 2)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return start_logits, end_logits