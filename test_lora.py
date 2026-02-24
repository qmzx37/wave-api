import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = r"C:\llm\hf\Meta-Llama-3-8B-Instruct"
ADAPTER = r"C:\llm\lora_8axis_adapter_v2"

print("BASE:", BASE)
print("ADAPTER:", ADAPTER)

tokenizer = AutoTokenizer.from_pretrained(ADAPTER, use_fast=True)

base = AutoModelForCausalLM.from_pretrained(
    BASE,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

model = PeftModel.from_pretrained(base, ADAPTER)
model.eval()

prompt = "친구랑 싸워서 힘들어. 내 감정 8축(F,A,D,J,C,G,T,R)을 0~1 실수로 JSON만 출력해. 다른 말 금지."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False
    )

text = tokenizer.decode(out[0], skip_special_tokens=True)
print("\n=== RAW OUTPUT ===")
print(text)
