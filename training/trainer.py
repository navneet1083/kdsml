import torch

class Trainer:
    def __init__(self, teacher_model, student_model, loss_fn, optimizer, device):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device

    def _train_step(self, input_ids, attention_mask, labels):
        self.student_model.train()
        self.teacher_model.eval()

        input_ids, attention_mask, labels = input_ids.to(self.device), attention_mask.to(self.device), labels.to(self.device)

        with torch.no_grad():
            teacher_outputs = self.teacher_model(input_ids, attention_mask)

        student_outputs = self.student_model(input_ids)
        loss = self.loss_fn(student_outputs, teacher_outputs, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def single_gpu_train_step(self, input_ids, attention_mask, labels):
        self.student_model.train()
        self.teacher_model.eval()

        input_ids, attention_mask, labels = (
            input_ids.to(self.device),
            attention_mask.to(self.device),
            labels.to(self.device)
        )

        with torch.no_grad():
            teacher_outputs = self.teacher_model(input_ids=input_ids, attention_mask=attention_mask)

        # Access the last_hidden_state from the teacher model's output
        teacher_hidden_states = teacher_outputs['last_hidden_state']

        # Pass both input_ids and attention_mask to the student model
        student_outputs = self.student_model(input_ids=input_ids, attention_mask=attention_mask)

        # Compute loss
        loss = self.loss_fn(student_outputs, teacher_hidden_states, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # print("Student Outputs Shape:", student_outputs.shape)
        # print("Teacher Hidden States Shape:", teacher_hidden_states.shape)
        # print("Labels Shape:", labels.shape)

        return loss.item()

    def train_step(self, input_ids, attention_mask, labels):
        self.student_model.train()
        self.teacher_model.eval()

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        labels = labels.to(self.device)

        with torch.no_grad():
            teacher_outputs = self.teacher_model(input_ids=input_ids, attention_mask=attention_mask)

        teacher_hidden_states = teacher_outputs['last_hidden_state']

        # Forward pass through the student model
        student_outputs = self.student_model(input_ids=input_ids, attention_mask=attention_mask)

        # Compute loss using the teacher's hidden states
        loss = self.loss_fn(student_outputs, teacher_hidden_states, labels)

        # Instead of calling backward and optimizer step here,
        # simply return the loss tensor for the caller to handle.
        return loss