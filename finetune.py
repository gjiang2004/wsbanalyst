import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID       = os.getenv("WSB_BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
DATA_FILE      = os.getenv("WSB_TRAINING_DATA", "wsb_training_data.jsonl")
OUTPUT_DIR     = os.getenv("WSB_FINETUNED_DIR", "wsb-mistral-finetuned2")
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "512"))
EVAL_SPLIT     = float(os.getenv("EVAL_SPLIT", "0.05"))
NUM_PROC       = max(1, (os.cpu_count() or 2) - 1)

# ── TF32 — free speedup on Ampere GPUs (3090, 4090, etc.) ────────────────────
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True

# ── Tokenizer ─────────────────────────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "right"

# ── 4-bit quantisation (QLoRA) ────────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# ── Base model ────────────────────────────────────────────────────────────────
print("Loading model in 4-bit...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
)
model = prepare_model_for_kbit_training(model)

# ── LoRA adapter ──────────────────────────────────────────────────────────────
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj",
        "o_proj", "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── Dataset ───────────────────────────────────────────────────────────────────
print("Loading dataset...")
raw = load_dataset("json", data_files=DATA_FILE, split="train")
print(f"Total examples: {len(raw)}")

# pre-tokenize across all cores before training starts so the GPU is never
# starved by single-threaded tokenization during the training loop
print(f"Pre-tokenizing dataset across {NUM_PROC} cores...")
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding=False,
    )

raw = raw.map(
    tokenize,
    batched=True,
    num_proc=NUM_PROC,
    remove_columns=["text"],
    desc="Tokenizing",
)

split      = raw.train_test_split(test_size=EVAL_SPLIT, seed=42)
train_data = split["train"]
eval_data  = split["test"]
print(f"Train: {len(train_data)}  |  Eval: {len(eval_data)}")

# ── Training config ───────────────────────────────────────────────────────────
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,

    # epochs / batch
    num_train_epochs=float(os.getenv("NUM_TRAIN_EPOCHS", "2")),
    per_device_train_batch_size=int(os.getenv("TRAIN_BATCH_SIZE", "8")),
    gradient_accumulation_steps=int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "2")),

    # optimiser
    learning_rate=float(os.getenv("LEARNING_RATE", "2e-4")),
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    optim="paged_adamw_8bit",

    # precision
    fp16=False,
    bf16=True,

    # sequence / packing
    max_length=MAX_SEQ_LENGTH,
    dataset_text_field=None,            # already tokenized, skip re-processing
    packing=True,

    # CPU dataloader
    dataloader_num_workers=NUM_PROC,
    dataloader_pin_memory=True,
    dataloader_prefetch_factor=4,       # each worker pre-queues 4 batches ahead

    # logging / saving
    logging_steps=20,
    eval_strategy="steps",
    eval_steps=100,
    save_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",

    report_to="none",
)

# ── Trainer ───────────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_data,
    eval_dataset=eval_data,
    processing_class=tokenizer,
)

# ── Train ─────────────────────────────────────────────────────────────────────
print("Starting training...")
trainer.train()

# ── Save ──────────────────────────────────────────────────────────────────────
print("Saving model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Done -> {OUTPUT_DIR}")