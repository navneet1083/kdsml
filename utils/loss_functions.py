import torch
import torch.nn as nn


class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.5, hidden_size=1024, vocab_size=30522):
        """
        Update the hidden_size to 1024 to match the teacher model's output.
        """
        super(KnowledgeDistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
        self.projection = nn.Linear(hidden_size, vocab_size)  # Projection layer now uses hidden_size=1024

    def forward(self, student_logits, teacher_hidden_states, labels):
        # Reshape teacher_hidden_states for projection
        batch_size, sequence_length, hidden_size = teacher_hidden_states.shape
        teacher_hidden_states_flat = teacher_hidden_states.view(-1, hidden_size)

        # Project teacher hidden states to match student logits shape (vocab_size)
        teacher_logits_flat = self.projection(teacher_hidden_states_flat)
        teacher_logits = teacher_logits_flat.view(batch_size, sequence_length, -1)

        # Soften probabilities for knowledge distillation
        soft_teacher = torch.softmax(teacher_logits / self.temperature, dim=-1)
        soft_student = torch.log_softmax(student_logits / self.temperature, dim=-1)

        # Compute KL divergence loss
        kd_loss = self.kl_div(
            soft_student.view(-1, soft_student.size(-1)),
            soft_teacher.view(-1, soft_teacher.size(-1))
        ) * (self.temperature ** 2)

        # Compute cross-entropy loss
        ce_loss = self.ce_loss(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1)
        )

        # Combine the losses
        return self.alpha * ce_loss + (1 - self.alpha) * kd_loss
