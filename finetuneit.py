from models.student_model import StudentModel
from fine_tuning.finetunestudentmodel import FineTuneStudentModel
from fine_tuning.finetunetrainer import FineTuneTrainer
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from datasets import load_dataset

from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

import os
from tqdm import tqdm


def main():
    student_model_name = "checkpoint.pth"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load GLUE SST-2 dataset using Hugging Face Datasets
    raw_dataset = load_dataset("glue", "sst2")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def preprocess_function(examples):
        return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=128)

    tokenized_dataset = raw_dataset.map(preprocess_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["sentence", "idx"])
    tokenized_dataset.set_format("torch")
    train_loader = DataLoader(tokenized_dataset["train"], batch_size=512, shuffle=True)
    val_loader = DataLoader(tokenized_dataset["validation"], batch_size=512)
    test_loader = DataLoader(tokenized_dataset["test"], batch_size=512)

    num_labels = 2  # SST-2 is binary classification

    # -------------------------------
    # 4a. Load Pre-trained Student Model
    # -------------------------------
    student_base = StudentModel().to(device)
    # Load pre-training weights from "student_model.pth"
    student_checkpoint = torch.load(student_model_name, map_location=device)
    student_base.load_state_dict(student_checkpoint['student_model_state_dict'])

    # Wrap the student model for fine-tuning
    fine_tune_student = FineTuneStudentModel(student_base, num_labels=num_labels).to(device)

    # -------------------------------
    # 4b. Load a Real BERT Model for Benchmarking
    # -------------------------------
    teacher_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels).to(device)

    # -------------------------------
    # 4c. Create Optimizers and Trainers
    # -------------------------------
    student_optimizer = AdamW(fine_tune_student.parameters(), lr=2e-5)
    teacher_optimizer = AdamW(teacher_model.parameters(), lr=2e-5)
    student_trainer = FineTuneTrainer(fine_tune_student, student_optimizer, device)
    teacher_trainer = FineTuneTrainer(teacher_model, teacher_optimizer, device)

    # -------------------------------
    # 4d. Checkpointing Setup: Load existing checkpoint if available
    # -------------------------------
    student_ckpt_path = "fine_tune_student_ckpt.pth"
    teacher_ckpt_path = "fine_tune_teacher_ckpt.pth"
    start_epoch = 0
    student_train_losses, student_val_losses, student_val_accuracies = [], [], []
    teacher_train_losses, teacher_val_losses, teacher_val_accuracies = [], [], []

    if os.path.exists(student_ckpt_path):
        ckpt = torch.load(student_ckpt_path, map_location=device)
        fine_tune_student.load_state_dict(ckpt["model_state_dict"])
        student_optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        student_train_losses = ckpt.get("train_losses", [])
        student_val_losses = ckpt.get("val_losses", [])
        student_val_accuracies = ckpt.get("val_accuracies", [])
        print(f"Resuming student fine-tuning from epoch {start_epoch}.")

    if os.path.exists(teacher_ckpt_path):
        ckpt = torch.load(teacher_ckpt_path, map_location=device)
        teacher_model.load_state_dict(ckpt["model_state_dict"])
        teacher_optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = max(start_epoch, ckpt["epoch"] + 1)
        teacher_train_losses = ckpt.get("train_losses", [])
        teacher_val_losses = ckpt.get("val_losses", [])
        teacher_val_accuracies = ckpt.get("val_accuracies", [])
        print(f"Resuming teacher fine-tuning from epoch {start_epoch}.")

    num_epochs = 5  # Adjust as needed.
    patience = 3
    best_student_val_loss = float("inf")
    best_teacher_val_loss = float("inf")
    student_patience_counter = 0
    teacher_patience_counter = 0

    # -------------------------------
    # 4e. Fine-Tuning Loop: Train and Validate
    # -------------------------------
    for epoch in range(start_epoch, num_epochs):
        fine_tune_student.train()
        teacher_model.train()
        total_student_loss = 0.0
        total_teacher_loss = 0.0
        total_student_acc = 0.0
        total_teacher_acc = 0.0
        student_batches = 0
        teacher_batches = 0

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}",
                    unit="batch")

        for i, batch in pbar:
            s_loss, s_acc = student_trainer.train_step(batch)
            t_loss, t_acc = teacher_trainer.train_step(batch)
            total_student_loss += s_loss
            total_teacher_loss += t_loss
            total_student_acc += s_acc
            total_teacher_acc += t_acc
            student_batches += 1
            teacher_batches += 1
            pbar.set_postfix({
                "Student Loss": f"{s_loss:.4f}",
                "Teacher Loss": f"{t_loss:.4f}"
            })

        avg_student_train_loss = total_student_loss / student_batches
        avg_teacher_train_loss = total_teacher_loss / teacher_batches
        avg_student_train_acc = total_student_acc / student_batches
        avg_teacher_train_acc = total_teacher_acc / teacher_batches

        student_train_losses.append(avg_student_train_loss)
        teacher_train_losses.append(avg_teacher_train_loss)

        student_val_loss, student_val_acc = student_trainer.validate(val_loader)
        teacher_val_loss, teacher_val_acc = teacher_trainer.validate(val_loader)
        student_val_losses.append(student_val_loss)
        teacher_val_losses.append(teacher_val_loss)
        student_val_accuracies.append(student_val_acc)
        teacher_val_accuracies.append(teacher_val_acc)

        print(f"Epoch {epoch + 1}:")
        print(
            f"  Student -> Train Loss: {avg_student_train_loss:.4f}, Train Acc: {avg_student_train_acc:.4f} | Val Loss: {student_val_loss:.4f}, Val Acc: {student_val_acc:.4f}")
        print(
            f"  Teacher -> Train Loss: {avg_teacher_train_loss:.4f}, Train Acc: {avg_teacher_train_acc:.4f} | Val Loss: {teacher_val_loss:.4f}, Val Acc: {teacher_val_acc:.4f}")

        # Early Stopping for Student
        if student_val_loss < best_student_val_loss:
            best_student_val_loss = student_val_loss
            student_patience_counter = 0
        else:
            student_patience_counter += 1

        # Early Stopping for Teacher
        if teacher_val_loss < best_teacher_val_loss:
            best_teacher_val_loss = teacher_val_loss
            teacher_patience_counter = 0
        else:
            teacher_patience_counter += 1

        # Save checkpoints
        torch.save({
            "epoch": epoch,
            "model_state_dict": fine_tune_student.state_dict(),
            "optimizer_state_dict": student_optimizer.state_dict(),
            "train_losses": student_train_losses,
            "val_losses": student_val_losses,
            "val_accuracies": student_val_accuracies,
        }, student_ckpt_path)
        torch.save({
            "epoch": epoch,
            "model_state_dict": teacher_model.state_dict(),
            "optimizer_state_dict": teacher_optimizer.state_dict(),
            "train_losses": teacher_train_losses,
            "val_losses": teacher_val_losses,
            "val_accuracies": teacher_val_accuracies,
        }, teacher_ckpt_path)
        print(f"Checkpoint saved for epoch {epoch + 1}.")

        if student_patience_counter >= patience:
            print("Early stopping student fine-tuning.")
            break
        if teacher_patience_counter >= patience:
            print("Early stopping teacher fine-tuning.")
            break

    # -------------------------------
    # 4f. Final Evaluation and Saving
    # -------------------------------
    student_final_loss, student_final_acc = student_trainer.validate(val_loader)
    teacher_final_loss, teacher_final_acc = teacher_trainer.validate(val_loader)
    print(f"Final Student: Val Loss: {student_final_loss:.4f}, Val Acc: {student_final_acc:.4f}")
    print(f"Final Teacher: Val Loss: {teacher_final_loss:.4f}, Val Acc: {teacher_final_acc:.4f}")

    torch.save(fine_tune_student.state_dict(), "fine_tuned_student_model.pth")
    torch.save(teacher_model.state_dict(), "fine_tuned_teacher_model.pth")

    save_training_plot(student_train_losses, student_val_losses, student_val_accuracies, "student", epoch + 1)
    save_training_plot(teacher_train_losses, teacher_val_losses, teacher_val_accuracies, "teacher", epoch + 1)


def save_training_plot(train_losses, val_losses, val_accuracies, model_name, final_epoch):
    os.makedirs("plots", exist_ok=True)
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(14, 6))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Loss over Epochs')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', marker='o', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{model_name} Accuracy over Epochs')
    plt.legend()

    plt.tight_layout()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"plots/{model_name}_finetune_epoch{final_epoch}_{timestamp}.png"
    plt.savefig(filename)
    print(f"Training metrics plot saved as: {filename}")



if __name__ == "__main__":
    main()
