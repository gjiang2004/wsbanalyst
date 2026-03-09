import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
FINETUNED_DIR = "wsb-mistral-finetuned"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(FINETUNED_DIR)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
)

print("Loading fine-tuned weights...")
model = PeftModel.from_pretrained(model, FINETUNED_DIR)
model.eval()

# Conversation history
history = []

def build_prompt(history, user_message):
    """Build full conversation prompt with history"""
    prompt = ""
    for turn in history:
        prompt += f"<s>[INST] {turn['user']} [/INST] {turn['bot']}</s>"
    prompt += f"<s>[INST] {user_message.strip()} [/INST]"
    return prompt

def chat(user_message, max_new_tokens=150):
    global history

    prompt = build_prompt(history, user_message)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.85,
            do_sample=True,
            top_p=0.92,
            repetition_penalty=1.15,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

    # Save to history (keep last 6 turns to avoid context overflow)
    history.append({"user": user_message, "bot": response})
    if len(history) > 6:
        history.pop(0)

    return response

print("✅ WSB Bot ready! Type your message (or 'quit' to exit)\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ("quit", "exit", "q"):
        break
    if user_input.lower() == "reset":
        history = []
        print("Conversation reset.\n")
        continue
    if not user_input:
        continue
    response = chat(user_input)
    print(f"\nWSB Bot: {response}\n")