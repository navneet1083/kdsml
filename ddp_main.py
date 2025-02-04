# ddp_main.py
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import your modules (make sure they are modified to work in a distributed context)
from models.teacher_model import TeacherModel
from models.student_model import StudentModel
from utils.loss_functions import KnowledgeDistillationLoss
from training.trainer import Trainer
from training.validator import Validator
from evaluation.evaluator import Evaluator
from utils.data_loader import DataLoader

def main():
    # Initialize the distributed process group (use NCCL backend for GPUs)
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = f"cuda:{local_rank}"

    # Create teacher and student models
    teacher_model = TeacherModel().to(device)
    student_model = StudentModel().to(device)

    # Wrap student model with DDP; teacher is used only for inference so we keep it unwrapped
    student_model = DDP(student_model, device_ids=[local_rank], output_device=local_rank)

    # Loss function: update hidden_size to match teacher (1024) and move to the correct device
    loss_fn = KnowledgeDistillationLoss(hidden_size=1024).to(device)

    # Optimizer (only updating the student model parameters)
    optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-4)

    # Mixed precision scaler
    scaler = GradScaler()

    # Create trainer and validator (update these classes if needed to support AMP if you want internal handling)
    trainer = Trainer(teacher_model, student_model, loss_fn, optimizer, device)
    validator = Validator(student_model, loss_fn, device)
    evaluator = Evaluator(student_model.module, teacher_model.tokenizer, device)  # use student_model.module for the underlying model

    # Prepare DataLoader using DistributedSampler for the training set
    data_loader_obj = DataLoader(dataset_name='wikitext')
    train_loader = data_loader_obj.get_dataloader(split='train', batch_size=32)
    # Replace the sampler with DistributedSampler so that each process gets a distinct subset
    train_loader.sampler = DistributedSampler(train_loader.dataset)
    val_loader = DataLoader(dataset_name='wikitext').get_dataloader(split='validation', batch_size=32)


    eval_loader = DataLoader(dataset_name='squad').get_dataloader(split='validation', batch_size=16)

    num_epochs = 10
    # Containers for metrics for visualization
    train_losses = []
    val_losses = []
    val_accuracies = []

    # Main training loop
    for epoch in range(num_epochs):
        student_model.train()
        total_train_loss = 0.0
        # Set epoch for sampler for reproducible shuffling
        train_loader.sampler.set_epoch(epoch)
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")

        for i, batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            # Mixed precision forward pass
            with autocast():
                loss = trainer.train_step(input_ids, attention_mask, labels)
            # Scale the loss and perform backpropagation
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()
            pbar.set_postfix({
                'Iter': f"{i+1}/{len(train_loader)}",
                'Loss': f"{loss.item():.4f}"
            })

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation step (assume validator.validate returns (loss, accuracy))
        val_loss, val_accuracy = validator.validate(val_loader)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Only the main process should print and save checkpoints
        if local_rank == 0:
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  Val Acc:    {val_accuracy:.2f}%")

            # Save checkpoint (including student model, optimizer state, and metrics)
            checkpoint = {
                'epoch': epoch,
                'student_model_state_dict': student_model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_accuracies': val_accuracies
            }
            torch.save(checkpoint, "checkpoint.pth")
            print(f"Checkpoint saved for epoch {epoch+1}\n")

    # After training, perform evaluation on SQuAD
    predictions = evaluator.evaluate(eval_loader)
    if local_rank == 0:
        torch.save(student_model.module.state_dict(), 'student_model.pth')

    # Visualization of training metrics (only run on main process)
    if local_rank == 0:
        plot_metrics(train_losses, val_losses, val_accuracies)

    # Cleanup the distributed process group
    dist.destroy_process_group()

def plot_metrics(train_losses, val_losses, val_accuracies):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(14, 6))
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, val_losses, label='Val Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label='Val Accuracy', marker='o', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy over Epochs')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
