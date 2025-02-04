# main.py
from models.teacher_model import TeacherModel
from models.student_model import StudentModel
from utils.loss_functions import KnowledgeDistillationLoss
from training.trainer import Trainer
from training.validator import Validator
from evaluation.evaluator import Evaluator
from utils.data_loader import load_squad_data

# Initialize models, loss, and optimizer
teacher_model = TeacherModel().to('cuda')
student_model = StudentModel().to('cuda')
loss_fn = KnowledgeDistillationLoss()
optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-4)

# Initialize trainer and validator
trainer = Trainer(teacher_model, student_model, loss_fn, optimizer, 'cuda')
validator = Validator(student_model, loss_fn, 'cuda')

# Load and preprocess data
train_data = load_squad_data('train-v2.0.json')
val_data = load_squad_data('dev-v2.0.json')

# Training loop
for epoch in range(10):
    train_loss = trainer.train_step(train_data)
    val_loss = validator.validate(val_data)
    print(f"Epoch {epoch+1}: Train Loss = {train_loss}, Val Loss = {val_loss}")