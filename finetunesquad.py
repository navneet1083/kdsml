from models.student_model import StudentModel
from fine_tuning.finetunestudentqa import FineTuneStudentQA
from fine_tuning.qatrainer import QA_Trainer
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForQuestionAnswering, AdamW
from datasets import load_dataset
import json
import os
from tqdm import tqdm

import datetime
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load SQuAD dataset
    squad_dataset = load_dataset("squad")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    def preprocess_squad(examples):
        tokenized = tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=384,
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length"
        )
        return tokenized
    tokenized_dataset = squad_dataset.map(preprocess_squad, batched=True, remove_columns=squad_dataset["train"].column_names)
    # For demonstration, add dummy start/end positions (set to 0)
    def add_dummy_positions(example):
        example["start_positions"] = 0
        example["end_positions"] = 0
        return example
    tokenized_dataset = tokenized_dataset.map(add_dummy_positions)
    tokenized_dataset.set_format("torch")
    train_loader = DataLoader(tokenized_dataset["train"], batch_size=8, shuffle=True)
    val_loader = DataLoader(tokenized_dataset["validation"], batch_size=8)

    num_epochs = 3
    # --- Load Pre-trained Student Model for QA ---
    from models.student_model import StudentModel
    student_base = StudentModel(hidden_size=384, num_layers=6, vocab_size=30522)
    student_checkpoint = torch.load("student_model.pth", map_location=device)
    student_base.load_state_dict(student_checkpoint)
    fine_tune_student = FineTuneStudentQA(student_base).to(device)

    # --- Load Teacher Model for QA (Benchmark) ---
    teacher_model = BertForQuestionAnswering.from_pretrained("bert-base-uncased").to(device)

    # --- Create Optimizers and Trainers ---
    student_optimizer = AdamW(fine_tune_student.parameters(), lr=3e-5)
    teacher_optimizer = AdamW(teacher_model.parameters(), lr=3e-5)
    student_trainer = QA_Trainer(fine_tune_student, student_optimizer, device)
    teacher_trainer = QA_Trainer(teacher_model, teacher_optimizer, device)

    # --- Checkpointing Setup ---
    student_ckpt_path = "fine_tune_student_qa_ckpt.pth"
    teacher_ckpt_path = "fine_tune_teacher_qa_ckpt.pth"
    start_epoch = 0
    student_train_losses = []
    teacher_train_losses = []
    student_val_losses = []
    teacher_val_losses = []

    if os.path.exists(student_ckpt_path):
        ckpt = torch.load(student_ckpt_path, map_location=device)
        fine_tune_student.load_state_dict(ckpt["model_state_dict"])
        student_optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        student_train_losses = ckpt.get("train_losses", [])
        student_val_losses = ckpt.get("val_losses", [])
        print(f"Resuming student QA fine-tuning from epoch {start_epoch}.")
    if os.path.exists(teacher_ckpt_path):
        ckpt = torch.load(teacher_ckpt_path, map_location=device)
        teacher_model.load_state_dict(ckpt["model_state_dict"])
        teacher_optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = max(start_epoch, ckpt["epoch"] + 1)
        teacher_train_losses = ckpt.get("train_losses", [])
        teacher_val_losses = ckpt.get("val_losses", [])
        print(f"Resuming teacher QA fine-tuning from epoch {start_epoch}.")

    for epoch in range(start_epoch, num_epochs):
        fine_tune_student.train()
        teacher_model.train()
        total_student_loss = 0.0
        total_teacher_loss = 0.0
        student_batches = 0
        teacher_batches = 0

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoc h +1}/{num_epochs}", unit="batch")
        for i, batch in pbar:
            s_loss = student_trainer.train_step(batch)
            t_loss = teacher_trainer.train_step(batch)
            total_student_loss += s_loss
            total_teacher_loss += t_loss
            student_batches += 1
            teacher_batches += 1
            pbar.set_postfix({
                "Student Loss": f"{s_loss:.4f}",
                "Teacher Loss": f"{t_loss:.4f}"
            })

        avg_student_train_loss = total_student_loss / student_batches
        avg_teacher_train_loss = total_teacher_loss / teacher_batches
        student_train_losses.append(avg_student_train_loss)
        teacher_train_losses.append(avg_teacher_train_loss)

        student_val_loss = student_trainer.validate(val_loader)
        teacher_val_loss = teacher_trainer.validate(val_loader)
        student_val_losses.append(student_val_loss)
        teacher_val_losses.append(teacher_val_loss)

        print(f"Epoch {epoch +1}:")
        print(f"  Student QA -> Train Loss: {avg_student_train_loss:.4f} | Val Loss: {student_val_loss:.4f}")
        print(f"  Teacher QA -> Train Loss: {avg_teacher_train_loss:.4f} | Val Loss: {teacher_val_loss:.4f}")

        torch.save({
            "epoch": epoch,
            "model_state_dict": fine_tune_student.state_dict(),
            "optimizer_state_dict": student_optimizer.state_dict(),
            "train_losses": student_train_losses,
            "val_losses": student_val_losses,
        }, student_ckpt_path)
        torch.save({
            "epoch": epoch,
            "model_state_dict": teacher_model.state_dict(),
            "optimizer_state_dict": teacher_optimizer.state_dict(),
            "train_losses": teacher_train_losses,
            "val_losses": teacher_val_losses,
        }, teacher_ckpt_path)
        print(f"Checkpoint saved for epoch {epoch +1}.")

    student_final_loss = student_trainer.validate(val_loader)
    teacher_final_loss = teacher_trainer.validate(val_loader)
    print(f"Final Student QA: Val Loss: {student_final_loss:.4f}")
    print(f"Final Teacher QA: Val Loss: {teacher_final_loss:.4f}")

    torch.save(fine_tune_student.state_dict(), "fine_tuned_student_qa_model.pth")
    torch.save(teacher_model.state_dict(), "fine_tuned_teacher_qa_model.pth")
    save_training_plot(student_train_losses, student_val_losses, "student_qa", epoch +1)
    save_training_plot(teacher_train_losses, teacher_val_losses, "teacher_qa", epoch +1)

    metrics = {
        "train_losses": student_train_losses,
        "val_losses": student_val_losses
    }
    with open("squad_metrics.json", "w") as f:
        json.dump(metrics, f)
    print("SQuAD metrics saved.")

def save_training_plot(train_losses, val_losses, model_name, final_epoch):
    os.makedirs("plots", exist_ok=True)
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, label="Train Loss", marker="o")
    plt.plot(epochs, val_losses, label="Validation Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} QA Loss over Epochs")
    plt.legend()
    plt.tight_layout()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"plots/{model_name}_qa_finetune_epoch{final_epoch}_{timestamp}.png"
    plt.savefig(filename)
    print(f"Plot saved as: {filename}")

if __name__ == "__main__":
    main()