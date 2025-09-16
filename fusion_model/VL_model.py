
def load_vision_language_model(
    llm_name_or_path: str,
    vision_name_or_path: str = "openai/clip-vit-large-patch14",
    tokenizer_name_or_path: str = None,
    device_map: str = "auto",
    load_in_8bit: bool = False,
    load_in_half: bool = True,
    image_token: str = "<image>",
    freeze_vision: bool = True,
    freeze_llm: bool = False,
    use_fast_tokenizer: bool = False,
    use_safetensors: bool = False,
    projector_type: str = "mlp",  # "mlp" or "perceiver"
    num_latents: int = 64,
    depth: int = 4,
    heads: int = 8,
    ff_mult: int = 4,
    dropout: float = 0.0,
):
    """Construct a fused vision-language model (vision encoder + projector + Qwen LLM).

    Returns: (fused_model, llm_tokenizer, vision_processor)
    """
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        CLIPVisionModel,
        CLIPImageProcessor,
    )
    from mm.fusion_model import VisionLanguageModel, VisionProjector, VisionPerceiverResampler

    if not tokenizer_name_or_path:
        tokenizer_name_or_path = llm_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        use_fast=use_fast_tokenizer,
        padding_side="left",
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        if tokenizer.unk_token:
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.pad_token_id = tokenizer.unk_token_id
        elif tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            raise ValueError("Tokenizer lacks pad token and eos token.")

    # Load LLM
    if load_in_8bit:
        llm = AutoModelForCausalLM.from_pretrained(
            llm_name_or_path,
            device_map=device_map,
            load_in_8bit=True,
            trust_remote_code=True,
        )
    else:
        llm = AutoModelForCausalLM.from_pretrained(
            llm_name_or_path,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True,
            use_safetensors=use_safetensors,
        )
        if torch.cuda.is_available() and load_in_half:
            llm = llm.half()
    llm.eval()

    # Load vision encoder and processor
    vision = CLIPVisionModel.from_pretrained(
        vision_name_or_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
        device_map=device_map,
    )
    vision_processor = CLIPImageProcessor.from_pretrained(vision_name_or_path)

    # Build projector
    llm_hidden = getattr(llm.config, "hidden_size")
    vision_hidden = getattr(vision.config, "hidden_size")
    if projector_type == "mlp":
        projector = VisionProjector(vision_hidden, llm_hidden)
    elif projector_type == "perceiver":
        projector = VisionPerceiverResampler(
            vision_hidden, llm_hidden,
            num_latents=num_latents,
            depth=depth,
            heads=heads,
            ff_mult=ff_mult,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown projector_type: {projector_type}")

    # Instantiate fused model
    fused = VisionLanguageModel(
        vision_encoder=vision,
        llm=llm,
        llm_tokenizer=tokenizer,
        projector=projector,
        image_token=image_token,
        freeze_vision=freeze_vision,
        freeze_llm=freeze_llm,
    )

    # Place projector on same device as LLM embeddings
    try:
        device = next(llm.parameters()).device
        fused.projector.to(device)
    except StopIteration:
        pass

    return fused, tokenizer, vision_processor

