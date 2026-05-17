import os
import sys

# Ensure repo root is on the path so echo_dsrn / echo_hybrid are importable
# regardless of where the script is invoked from.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "ethicalabs/Echo-DSRN-114M-v0.1.2"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", trust_remote_code=True)

inputs = tokenizer("The theory of predictive coding suggests that", return_tensors="pt").to(
    model.device
)
outputs = model.generate(**inputs, max_new_tokens=50)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
