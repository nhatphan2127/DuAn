"""Explicit NLLB LoRA construction and pinned adapter reload contract."""

import json
from pathlib import Path

import torch


def apply_lora(model, config):
    from peft import LoraConfig, TaskType, get_peft_model

    spec = config["model"]
    options = spec["lora"]
    targets = options["target_modules"]
    if model.config.model_type != "m2m_100":
        raise ValueError("LoRA experiment requires NLLB's M2M100 architecture")
    names = [name for name, module in model.named_modules()
             if name.rsplit(".", 1)[-1] in targets and isinstance(module, torch.nn.Linear)]
    expected = [f"model.{side}.layers.{layer}.{attention}.{target}"
                for side, count in (("encoder", model.config.encoder_layers), ("decoder", model.config.decoder_layers))
                for layer in range(count)
                for attention in (("self_attn",) if side == "encoder" else ("self_attn", "encoder_attn"))
                for target in targets]
    if not names or set(names) != set(expected):
        raise ValueError("LoRA target inventory differs from encoder/decoder attention projections")
    model = get_peft_model(model, LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM,
        r=options["r"], lora_alpha=options["alpha"], lora_dropout=options["dropout"],
        target_modules=targets, bias="none", modules_to_save=None,
        revision=spec["revision"]))
    model.peft_config["default"].base_model_name_or_path = spec["name_or_path"]
    trainable = {name: p.numel() for name, p in model.named_parameters() if p.requires_grad}
    if not trainable or any(".lora_A." not in name and ".lora_B." not in name for name in trainable):
        raise ValueError("Only LoRA A/B parameters may be trainable; full fine-tuning refused")
    total = sum(p.numel() for p in model.parameters())
    model.lora_inventory = {"modules": sorted(names), "module_count": len(names),
        "trainable_parameters": sum(trainable.values()), "total_parameters": total,
        "trainable_fraction": sum(trainable.values()) / total, "parameters": trainable,
        "adapter": options, "base_model": spec["name_or_path"], "revision": spec["revision"]}
    print(json.dumps(model.lora_inventory, indent=2), flush=True)
    return model


def verify_adapter(directory, spec):
    directory = Path(directory)
    adapter = json.loads((directory / "adapter_config.json").read_text(encoding="utf-8"))
    base = json.loads((directory / "base_model.json").read_text(encoding="utf-8"))
    expected = spec["lora"]
    if (base["model"] != spec or adapter["base_model_name_or_path"] != spec["name_or_path"]
        or adapter["revision"] != spec["revision"] or adapter["peft_type"] != "LORA"
        or adapter["task_type"] != "SEQ_2_SEQ_LM" or adapter["r"] != expected["r"]
        or adapter["lora_alpha"] != expected["alpha"] or adapter["lora_dropout"] != expected["dropout"]
        or set(adapter["target_modules"]) != set(expected["target_modules"])
        or adapter["bias"] != "none" or adapter.get("modules_to_save")):
        raise ValueError("Adapter/base revision or LoRA experiment configuration mismatch")
    if not (directory / "adapter_model.safetensors").is_file():
        raise ValueError("Missing adapter safetensors")
    return adapter
