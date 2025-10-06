import math
import time

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from specalign.utils import onpolicy_generation

MAX_INPUT_SEQ_LENGTH = 512
LR = 2e-4
GRAD_ACCUM_STEPS = 16
WARMUP_STEPS = 100
MAX_STEPS = 2_000
K_MAX = 4
KD_TEMP = 1


def kd_loss_from_logits(logits_s, logits_t, targets, mask, T=1.0):
    logp_s = torch.log_softmax(logits_s / T, dim=-1)
    p_t = torch.softmax(logits_t / T, dim=-1)
    kd_token = torch.nn.functional.kl_div(logp_s, p_t, reduction="none") * (T * T)
    kd_seq = kd_token.sum(dim=-1)
    loss = (kd_seq[mask]).mean()

    # Calculate agreement rate
    agree = (logits_s.argmax(-1) == logits_t.argmax(-1)) * mask
    agree_rate = agree.float().sum() / mask.float().sum().clamp_min(1)

    return loss, agree_rate.item()


def build_masks_and_targets(sequence, prompt_lengths):
    B, L = sequence.shape
    targets = sequence[:, 1:]
    drop_idx = torch.arange(L - 1, device=sequence.device).view(1, -1).expand(B, -1)
    mask = drop_idx >= (prompt_lengths.view(-1, 1) - 1)
    return targets, mask


def tokenize_function(examples, tokenizer, max_length):
    prompt = [
        {"role": "system", "content": "You are a concise, factual assistant."},
        {
            "role": "user",
            "content": f"Summarize this in 1 sentence {examples['article']}",
        },
    ]

    end_marker = "\n\nSummary:"
    end_ids = tokenizer(end_marker, add_special_tokens=False)["input_ids"]

    tokenized_prompt = tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        padding="max_length",
        truncation=True,
        max_length=max_length - len(end_ids),
        return_tensors=None,
    )

    return {"input_ids": tokenized_prompt["input_ids"] + end_ids}


def main():
    # =================
    # Setup
    # =================
    student_id = "Qwen/Qwen2.5-0.5B-Instruct"
    teacher_id = "Qwen/Qwen2.5-1.5B-Instruct"

    device = "cuda:1"
    attn_implementation = "flash_attention_2"

    # Load tokenizer from student (should be the same, but generating based on student)
    tokenizer = AutoTokenizer.from_pretrained(student_id)
    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.pad_token_id

    # Load Student and set pad/eos tokens
    student = AutoModelForCausalLM.from_pretrained(
        student_id,
        dtype=torch.bfloat16,
        memory_map=device,
        attn_implementation=attn_implementation,
    )
    student.config.pad_token_id = pad_token_id
    student.config.eos_token_id = eos_token_id

    # Load Teacher and set pad/eos tokens
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_id,
        dtype=torch.bfloat16,
        memory_map=device,
        attn_implementation=attn_implementation,
    )
    teacher.config.pad_token_id = pad_token_id
    teacher.config.eos_token_id = eos_token_id

    # ================
    # Data
    # ================
    ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split="train")

    tokenized_dataset = ds.map(
        tokenize_function,
        fn_kwargs={"tokenizer": tokenizer, "max_length": MAX_INPUT_SEQ_LENGTH},
        batched=False,
        remove_columns=ds.column_names,
    )

    # Configures dataset to return torch tensors when indexing
    tokenized_dataset = tokenized_dataset.with_format(
        type="torch",
        columns=["input_ids", "attention_mask"],
    )

    dl = DataLoader(tokenized_dataset, batch_size=10, shuffle=True, drop_last=True)

    # ================
    # Optimizer
    # ================
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=LR,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    # ================
    # Scheduler
    # ================
    steps_per_epoch = math.ceil(len(ds) / GRAD_ACCUM_STEPS)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=MAX_STEPS,
    )

    global_step = 0
    running = {"loss": 0.0, "agree": 0.0, "count": 0}

    student.train()
    time_start = time.time()

    # ================
    # Training Loop
    # ================
    while global_step < MAX_STEPS:
        for batch in dl:
            with torch.no_grad():
                # Generate sequence(s)
                sequences = onpolicy_generation(
                    model=student,
                    input_ids=batch["input_ids"].to(device),
                    k=K_MAX,
                )

            # Evaluate (Teacher & Student), Drop final logit (no target for last token)
            logits_s = student(input_ids=sequences).logits[:, :-1, :]
            with torch.no_grad():
                logits_t = teacher(input_ids=sequences).logits[:, :-1, :]

            # Generate target and masks
            prompt_lengths = batch["attention_mask"].sum(dim=1).to(device)
            targets, mask = build_masks_and_targets(sequences, prompt_lengths)

            # Calculate KL divergence loss
            kd, agree = kd_loss_from_logits(
                logits_s, logits_t, targets, mask, T=KD_TEMP
            )
            loss = kd / GRAD_ACCUM_STEPS

            loss.backward()

            # Track metrics
            running["loss"] += kd.item()
            running["agree"] += agree
            running["count"] += 1

            if (global_step + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Log progress
            if (global_step + 1) % 50 == 0:
                avg_loss = running["loss"] / running["count"]
                avg_agree = running["agree"] / running["count"]
                dt = time.time() - time_start
                print(
                    f"[step {global_step+1}] kd={avg_loss:.3f} agree={avg_agree:.3f} k={K_MAX} dt={dt:.1f}s"
                )
                running = {"loss": 0.0, "agree": 0.0, "count": 0}
                time_start = time.time()

            global_step += 1
            if global_step >= MAX_STEPS:
                break


if __name__ == "__main__":
    main()
