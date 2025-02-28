
class FineTuneStudentModel(nn.Module):
    def __init__(self, student_model, num_labels=2):
        """
        Wrap the pre-trained student model (from pre-training) by adding a classification head.
        The student_model is assumed to have a forward_hidden method that returns hidden states.
        """
        super(FineTuneStudentModel, self).__init__()
        self.student_model = student_model
        # Assume the hidden size is stored in the embedding dimension.
        hidden_size = self.student_model.embedding.embedding_dim
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None):
        # Obtain hidden states from the student model (without the original language-model head)
        hidden_states = self.student_model.forward_hidden(input_ids, attention_mask)
        # Use the [CLS] token representation (assumed to be at position 0)
        cls_token = hidden_states[:, 0, :]
        logits = self.classifier(cls_token)
        return logits