import os
from typing import List
import re
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

torch.set_num_threads(4)

def get_device():
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    return device

DEVICE = get_device()

model_name = "ai4bharat/indictrans2-indic-indic-dist-320M"

#hugging face token
token = os.getenv("HF_TOKEN")
tokenizer = AutoTokenizer.from_pretrained(model_name,  token=token, trust_remote_code=True)

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16 if DEVICE in ("cuda", "mps") else torch.float32,
).to(DEVICE)

model.eval()

if DEVICE == "cpu":
    model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8 )

ip = IndicProcessor(inference=True)

def translate(sentences: List[str], src_lang: str, target_lang: str):
    batch = ip.preprocess_batch(
        sentences,  src_lang=src_lang,tgt_lang=target_lang,)

    inputs = tokenizer(
        batch,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = model.generate( **inputs,
            num_beams=1,  do_sample=False, max_new_tokens=256,)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return ip.postprocess_batch(decoded, lang=target_lang)
