from datasets import load_dataset
from transformers import AutoTokenizer
from auto_fp8 import AutoFP8ForCausalLM, BaseQuantizeConfig
from typing import Union, List, Tuple

pretrained_model_dir = "BAAI/bge-reranker-base"
quantized_model_dir = "/root/bge-reranker-base-fp8"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

# Load and tokenize 512 dataset samples for calibration of activation scales
ds = load_dataset("mgoin/ultrachat_2k", split="train_sft").select(range(1))
examples = [tokenizer.apply_chat_template(batch["messages"], tokenize=False) for batch in ds]
sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]] = \
    [("hello world", "nice to meet you"), ("head north", "head south")]
examples = sentence_pairs
examples = tokenizer(examples, padding=True, truncation=True, return_tensors="pt").to("cuda")

# Define quantization config with static activation scales
quantize_config = BaseQuantizeConfig(quant_method="fp8", activation_scheme="static",
                                     ignore_patterns=["classifier.out_proj", "classifier.dense"])

# Load the model, quantize, and save checkpoint
model = AutoFP8ForCausalLM.from_pretrained(pretrained_model_dir, quantize_config)
model.quantize(examples)
model.save_quantized(quantized_model_dir)
