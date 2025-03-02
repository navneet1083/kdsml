from models.student_model import StudentModel
from fine_tuning.finetunestudentqa import FineTuneStudentQA
from fine_tuning.qatrainer import QA_Trainer
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertTokenizerFast, BertForQuestionAnswering, AdamW
from datasets import load_dataset
import json
import os
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import datetime
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()


def main():
    # Initialize distributed training
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = f"cuda:{local_rank}"

    student_model = "checkpoint.pth"

    # Load SQuAD dataset
    squad_dataset = load_dataset("squad")
    # Use BertTokenizerFast for offset mapping support
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

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
        # Remove and retrieve the overflow mapping.
        sample_mapping = tokenized.pop("overflow_to_sample_mapping", None)
        if sample_mapping is not None:
            tokenized["context"] = [examples["context"][i] for i in sample_mapping]
            tokenized["answers"] = [examples["answers"][i] for i in sample_mapping]
        else:
            tokenized["context"] = examples["context"]
            tokenized["answers"] = examples["answers"]
        return tokenized

    tokenized_dataset = squad_dataset.map(
        preprocess_squad,
        batched=True,
        remove_columns=["question", "title", "id"]
    )

    def add_dummy_positions(examples):
        num_examples = len(examples["input_ids"])
        return {
            "start_positions": [0 for _ in range(num_examples)],
            "end_positions": [0 for _ in range(num_examples)]
        }

    tokenized_dataset = tokenized_dataset.map(add_dummy_positions, batched=True)
    tokenized_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "offset_mapping", "context", "answers", "start_positions",
                 "end_positions"]
    )

    # Create Distributed Sampler for training DataLoader
    train_dataset = tokenized_dataset["train"]
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=128, sampler=train_sampler)

    val_loader = DataLoader(tokenized_dataset["validation"], batch_size=128)

    num_epochs = 10  # Adjust as needed for fine-tuning
    patience = 2  # Early stopping patience based on F1 improvement
    best_student_f1 = 0.0
    student_patience_counter = 0
    best_teacher_f1 = 0.0
    teacher_patience_counter = 0

    # -------------------------------
    # 4a. Load Pre-trained Student Model for QA Fine-Tuning
    # -------------------------------
    student_base = StudentModel().to(device)
    student_checkpoint = torch.load(student_model, map_location=device)
    student_base.load_state_dict(student_checkpoint['student_model_state_dict'])
    fine_tune_student = FineTuneStudentQA(student_base).to(device)
    fine_tune_student = DDP(fine_tune_student, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # -------------------------------
    # 4b. Load Teacher Model for QA Fine-Tuning (Benchmark)
    # -------------------------------
    teacher_model = BertForQuestionAnswering.from_pretrained("bert-base-uncased").to(device)
    teacher_model = DDP(teacher_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # -------------------------------
    # 4c. Create Optimizers and Trainers for QA
    # -------------------------------
    student_optimizer = AdamW(fine_tune_student.parameters(), lr=3e-5)
    teacher_optimizer = AdamW(teacher_model.parameters(), lr=3e-5)
    student_trainer = QA_Trainer(fine_tune_student, student_optimizer, device, tokenizer)
    teacher_trainer = QA_Trainer(teacher_model, teacher_optimizer, device, tokenizer)

    # -------------------------------
    # 4d. Checkpointing Setup: Load existing checkpoints if available
    # -------------------------------
    student_ckpt_path = "fine_tune_student_qa_ckpt.pth"
    teacher_ckpt_path = "fine_tune_teacher_qa_ckpt.pth"
    start_epoch = 0
    student_train_losses = []
    teacher_train_losses = []
    student_val_losses = []
    teacher_val_losses = []
    student_val_f1s = []
    teacher_val_f1s = []

    if os.path.exists(student_ckpt_path):
        ckpt = torch.load(student_ckpt_path, map_location=device)
        fine_tune_student.load_state_dict(ckpt["model_state_dict"])
        student_optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        student_train_losses = ckpt.get("train_losses", [])
        student_val_losses = ckpt.get("val_losses", [])
        student_val_f1s = ckpt.get("val_f1s", [])
        print(f"Resuming student QA fine-tuning from epoch {start_epoch}.")
    if os.path.exists(teacher_ckpt_path):
        ckpt = torch.load(teacher_ckpt_path, map_location=device)
        teacher_model.load_state_dict(ckpt["model_state_dict"])
        teacher_optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = max(start_epoch, ckpt["epoch"] + 1)
        teacher_train_losses = ckpt.get("train_losses", [])
        teacher_val_losses = ckpt.get("val_losses", [])
        teacher_val_f1s = ckpt.get("val_f1s", [])
        print(f"Resuming teacher QA fine-tuning from epoch {start_epoch}.")

    # -------------------------------
    # 4e. Fine-Tuning Loop with Early Stopping (based on F1)
    # -------------------------------
    for epoch in range(start_epoch, num_epochs):
        # Set epoch for DistributedSampler to shuffle data differently each epoch.
        train_sampler.set_epoch(epoch)
        fine_tune_student.train()
        teacher_model.train()
        total_student_loss = 0.0
        total_teacher_loss = 0.0
        student_batches = 0
        teacher_batches = 0

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}",
                    unit="batch")
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

        student_val_loss, student_val_f1 = student_trainer.validate(val_loader)
        teacher_val_loss, teacher_val_f1 = teacher_trainer.validate(val_loader)
        student_val_losses.append(student_val_loss)
        teacher_val_losses.append(teacher_val_loss)
        student_val_f1s.append(student_val_f1)
        teacher_val_f1s.append(teacher_val_f1)

        print(f"Epoch {epoch + 1}:")
        print(
            f"  Student QA -> Train Loss: {avg_student_train_loss:.4f} | Val Loss: {student_val_loss:.4f}, Val F1: {student_val_f1:.4f}")
        print(
            f"  Teacher QA -> Train Loss: {avg_teacher_train_loss:.4f} | Val Loss: {teacher_val_loss:.4f}, Val F1: {teacher_val_f1:.4f}")

        torch.save({
            "epoch": epoch,
            "model_state_dict": fine_tune_student.state_dict(),
            "optimizer_state_dict": student_optimizer.state_dict(),
            "train_losses": student_train_losses,
            "val_losses": student_val_losses,
            "val_f1s": student_val_f1s,
        }, student_ckpt_path)
        torch.save({
            "epoch": epoch,
            "model_state_dict": teacher_model.state_dict(),
            "optimizer_state_dict": teacher_optimizer.state_dict(),
            "train_losses": teacher_train_losses,
            "val_losses": teacher_val_losses,
            "val_f1s": teacher_val_f1s,
        }, teacher_ckpt_path)
        print(f"Checkpoint saved for epoch {epoch + 1}.")

        if student_val_f1 <= best_student_f1:
            student_patience_counter += 1
        else:
            best_student_f1 = student_val_f1
            student_patience_counter = 0
        if teacher_val_f1 <= best_teacher_f1:
            teacher_patience_counter += 1
        else:
            best_teacher_f1 = teacher_val_f1
            teacher_patience_counter = 0
        if student_patience_counter >= patience or teacher_patience_counter >= patience:
            print("Early stopping triggered based on F1 score.")
            break

    # Final Evaluation
    student_final_loss, student_final_f1 = student_trainer.validate(val_loader)
    teacher_final_loss, teacher_final_f1 = teacher_trainer.validate(val_loader)
    print(f"Final Student QA: Val Loss: {student_final_loss:.4f}, Val F1: {student_final_f1:.4f}")
    print(f"Final Teacher QA: Val Loss: {teacher_final_loss:.4f}, Val F1: {teacher_final_f1:.4f}")

    torch.save(fine_tune_student.state_dict(), "fine_tuned_student_qa_model.pth")
    torch.save(teacher_model.state_dict(), "fine_tuned_teacher_qa_model.pth")

    save_training_plot(student_train_losses, student_val_losses, student_val_f1s, "student_qa", epoch + 1)
    save_training_plot(teacher_train_losses, teacher_val_losses, teacher_val_f1s, "teacher_qa", epoch + 1)

    metrics = {
        "train_losses": student_train_losses,
        "val_losses": student_val_losses,
        "val_f1s": student_val_f1s
    }
    with open("squad_metrics.json", "w") as f:
        json.dump(metrics, f)
    print("SQuAD metrics saved.")

    # Clean up distributed group at the end.
    dist.destroy_process_group()


def save_training_plot(train_losses, val_losses, val_f1s, model_name, final_epoch):
    os.makedirs("plots", exist_ok=True)
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(14,6))
    plt.subplot(1,2,1)
    plt.plot(epochs, train_losses, label="Train Loss", marker="o")
    plt.plot(epochs, val_losses, label="Val Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} QA Loss over Epochs")
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(epochs, val_f1s, label="Val F1", marker="o", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title(f"{model_name} QA F1 over Epochs")
    plt.legend()
    plt.tight_layout()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"plots/{model_name}_qa_finetune_epoch{final_epoch}_{timestamp}.png"
    plt.savefig(filename)
    print(f"Plot saved as: {filename}")


if __name__ == "__main__":
    main()