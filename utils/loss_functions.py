# utils/loss_functions.py
import torch
import torch.nn as nn

class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, temperature=4.0):
        super(KnowledgeDistillationLoss, self).__init__()
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        soft_teacher = torch.softmax(teacher_logits / self.temperature, dim=-1)
        soft_student = torch.log_softmax(student_logits / self.temperature, dim=-1)
        kd_loss = self.kl_div(soft_student, soft_teacher) * (self.temperature ** 2)

        ce_loss = self.ce_loss(student_logits, labels)
        return kd_loss + ce_loss