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


def tokenize_function(examples, tokenizer, max_length):
    end_marker = ". Summary:"
    end_ids = tokenizer(end_marker, add_special_tokens=False)["input_ids"]

    prompt = [
        {"role": "system", "content": "You are a concise, factual assistant."},
        {"role": "user", "content": "Summarize the article in 1 sentence."},
        {"role": "user", "content": examples["article"]},
    ]

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

    return {"input_ids": [tokenized_prompt["input_ids"] + end_ids]}


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
        type="torch", columns=["input_ids"]
    )

    dl = DataLoader(tokenized_dataset, batch_size=10, shuffle=True, drop_last=True)

    # ================
    # Optimizer
    # ================
    optimizer = torch.optim.AdawmW(
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

    while global_step < MAX_STEPS:
        for batch in dl:

            input_ids = (
                tokenized_dataset[0]["input_ids"].unsqueeze(0).to(student.device)
            )

            with torch.no_grad():
                outputs = student.generate(
                    input_ids=input_ids,
                    max_new_tokens=K_MAX,
                    do_sample=False,
                    use_cache=True,
                    return_dict_in_generate=False,
                    output_scores=False,
                )


if __name__ == "__main__":
    main()
