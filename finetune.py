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
MODEL_ID       = "mistralai/Mistral-7B-Instruct-v0.3"
DATA_FILE      = "wsb_training_data.jsonl"
OUTPUT_DIR     = "wsb-mistral-finetuned2"
MAX_SEQ_LENGTH = 512
EVAL_SPLIT     = 0.05

# Leave one core for the main process — tune this to your machine.
# Run `python -c "import os; print(os.cpu_count())"` to see your core count.
DATALOADER_WORKERS = max(1, 12 - 1)

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

split      = raw.train_test_split(test_size=EVAL_SPLIT, seed=42)
train_data = split["train"]
eval_data  = split["test"]
print(f"Train: {len(train_data)}  |  Eval: {len(eval_data)}")
print(f"Dataloader workers: {DATALOADER_WORKERS}")

# ── Training config ───────────────────────────────────────────────────────────
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,

    # epochs / batch
    num_train_epochs=2,
    per_device_train_batch_size=8,      # increased from 4 — lower back to 4 if OOM
    gradient_accumulation_steps=2,

    # optimiser
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    optim="paged_adamw_8bit",

    # precision
    fp16=False,
    bf16=True,

    # sequence / packing
    max_length=MAX_SEQ_LENGTH,
    dataset_text_field="text",
    packing=True,                       # packs short examples to fill the full
                                        # context window — far fewer wasted GPU
                                        # cycles padding short WSB one-liners

    # CPU dataloader — preprocesses batches in parallel so GPU never starves
    dataloader_num_workers=DATALOADER_WORKERS,
    dataloader_pin_memory=True,         # pins CPU memory for faster GPU transfers

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
print(f"✅ Model saved to {OUTPUT_DIR}")