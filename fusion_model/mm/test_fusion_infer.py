import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, CLIPVisionModel, CLIPImageProcessor
from mm.fusion_model import VisionLanguageModel, VisionProjector, VisionPerceiverResampler

llm_id = "Qwen/Qwen2.5-Math-1.5B-Instruct"
vision_id = "openai/clip-vit-large-patch14"

tokenizer = AutoTokenizer.from_pretrained(llm_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

llm = AutoModelForCausalLM.from_pretrained(llm_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
vision = CLIPVisionModel.from_pretrained(vision_id, torch_dtype=torch.float16, device_map="auto")
processor = CLIPImageProcessor.from_pretrained(vision_id)

# Choose projector: MLP or Perceiver
use_perceiver = False
if use_perceiver:
    proj = VisionPerceiverResampler(vision.config.hidden_size, llm.config.hidden_size)
else:
    proj = VisionProjector(vision.config.hidden_size, llm.config.hidden_size)
model = VisionLanguageModel(vision, llm, tokenizer, proj)

# Prepare image tensor properly via processor
import numpy as np
dummy = (np.random.rand(vision.config.image_size, vision.config.image_size, 3) * 255).astype("uint8")
pix = processor(images=dummy, return_tensors="pt").pixel_values.to(next(llm.parameters()).device)

prompt = "Solve the problem in the image and output the final answer in \\boxed{}: <image>"
enc = tokenizer(prompt, return_tensors="pt")
input_ids = enc.input_ids.to(pix.device)

with torch.no_grad():
    out = model(
        input_ids=input_ids,
        images=pix,
        max_new_tokens=16,
        temperature=0.0,
    )

print(out)
