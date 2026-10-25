import os
import threading
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from chat_format import dtype_arg, ensure_chat_template, normalize_messages
from config import settings


def _use_4bit() -> bool:
    if not torch.cuda.is_available():
        return False

    if settings.TEXT_PRECISION == "4bit":
        return True

    if settings.TEXT_PRECISION == "fp16":
        return False

    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3

    return total_gb < 22


class TextEngine:
    """Mistral 7B Instruct with the optional marketing LoRA adapter."""

    def __init__(self):
        token = settings.HF_TOKEN or None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.quantized = _use_4bit()

        self.tokenizer = AutoTokenizer.from_pretrained(settings.TEXT_MODEL, token=token)
        ensure_chat_template(self.tokenizer)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs = {"token": token}

        if self.quantized:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            kwargs["device_map"] = {"": 0}
        elif self.device == "cuda":
            kwargs.update(dtype_arg(torch.float16))
            kwargs["device_map"] = {"": 0}
        else:
            kwargs.update(dtype_arg(torch.float32))

        model = AutoModelForCausalLM.from_pretrained(settings.TEXT_MODEL, **kwargs)

        self.adapter = None

        if settings.TEXT_ADAPTER and os.path.isfile(
            os.path.join(settings.TEXT_ADAPTER, "adapter_config.json")
        ):
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, settings.TEXT_ADAPTER)
            self.adapter = settings.TEXT_ADAPTER

            # Merging makes generation faster. A 4-bit base cannot be merged safely.
            if not self.quantized:
                model = model.merge_and_unload()

        model.eval()

        self.model = model
        self.lock = threading.Lock()

    def _prompt_ids(self, messages: list[dict], prefix: str):
        drop = 0

        while True:
            turns = normalize_messages(messages, drop_pairs=drop)
            text = self.tokenizer.apply_chat_template(
                turns, tokenize=False, add_generation_prompt=True
            ) + prefix
            ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids

            if ids.shape[1] <= settings.MAX_INPUT_TOKENS or len(turns) <= 1:
                return ids[:, -settings.MAX_INPUT_TOKENS:]

            drop += 1

    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int = 800,
        temperature: float = 0.7,
        top_p: float = 0.9,
        prefix: str = "",
    ) -> dict:
        started = time.time()
        max_new_tokens = max(16, min(max_new_tokens, settings.MAX_NEW_TOKENS_LIMIT))

        input_ids = self._prompt_ids(messages, prefix).to(self.model.device)

        options = {
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": 1.08,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if temperature > 0:
            options.update(do_sample=True, temperature=temperature, top_p=top_p)
        else:
            options.update(do_sample=False)

        with self.lock, torch.inference_mode():
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                **options,
            )

        new_tokens = output[0][input_ids.shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return {
            "text": (prefix + text).strip(),
            "tokens": int(new_tokens.shape[0]),
            "seconds": round(time.time() - started, 2),
        }
