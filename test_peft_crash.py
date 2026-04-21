import os
import sys

from gmsra.utils import load_model_and_tokenizer
print("Loading base model...")
model, tokenizer = load_model_and_tokenizer("Qwen/Qwen2.5-7B-Instruct")

lora_path = "outputs/phase1/best"
print("Loading PeftModel...")
try:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, lora_path)
    print("PeftModel loaded successfully.")
except Exception as e:
    print(f"PeftModel.from_pretrained failed: {e}")
    sys.exit(1)

from peft import LoraConfig, get_peft_model, TaskType
try:
    print("Running get_peft_model...")
    cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16)
    model = get_peft_model(model, cfg)
    print("get_peft_model succeeded")
except Exception as e:
    print(f"get_peft_model failed: {e}")
