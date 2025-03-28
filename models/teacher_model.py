import torch
from transformers import BertModel, BertTokenizer

class TeacherModel(torch.nn.Module):
    def __init__(self, pretrained_model='bert-large-uncased'):
        super(TeacherModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return {
            'last_hidden_state': outputs.last_hidden_state,  # Shape: (batch_size, sequence_length, hidden_size=768)
            'pooler_output': outputs.pooler_output
        }