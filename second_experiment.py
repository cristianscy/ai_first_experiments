import os
import torch
import transformers
from transformers import (
  AutoTokenizer,
  AutoModelForCausalLM,
  BitsAndBytesConfig
  pipeline
)

model_name="/home/cristian/development/ai/models/dolphin-2.7-mixtral-8x7b.Q4_K_M.gguf"

model_config = transformers.AutoConfig.from_pretrained(
   model_name,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

