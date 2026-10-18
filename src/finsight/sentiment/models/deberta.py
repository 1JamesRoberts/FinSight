"""DeBERTa-specific architecture and checked custom financial heads."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from finsight.sentiment.evaluation.metrics import LABELS


def masked_mean(hidden: Any, attention_mask: Any) -> Any:
    """Match the upstream mean over every non-padding token, including specials."""
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def build_model(config: Any) -> Any:
    """Reconstruct the published architecture without executing remote code."""
    import torch
    from transformers import AutoModel

    class MultiTaskDeberta(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = AutoModel.from_config(config, trust_remote_code=False)
            self.dropout = torch.nn.Dropout(0.1)
            self.cls_head = torch.nn.Linear(config.hidden_size, 3)
            self.reg_head = torch.nn.Linear(config.hidden_size, 1)
            self.config = config
            self.config.id2label = dict(enumerate(LABELS))

        def save_pretrained(self, directory: Path, *, safe_serialization: bool) -> None:
            from safetensors.torch import save_file

            if not safe_serialization:
                raise ValueError("derived weights require safe serialization")
            directory.mkdir(parents=True, exist_ok=True)
            save_file(self.state_dict(), str(directory / "model.safetensors"))
            self.config.save_pretrained(directory)

        def forward(
            self,
            input_ids: Any,
            attention_mask: Any,
            labels: Any = None,
            **kwargs: Any,
        ) -> Any:
            output = self.backbone(
                input_ids=input_ids, attention_mask=attention_mask, **kwargs
            )
            pooled = self.dropout(masked_mean(output.last_hidden_state, attention_mask))
            logits = self.cls_head(pooled)
            return SimpleNamespace(
                logits=logits,
                regression=self.reg_head(pooled).squeeze(-1),
                loss=torch.nn.functional.cross_entropy(logits, labels)
                if labels is not None
                else None,
            )

    return MultiTaskDeberta()


def load_tensor_state(model: Any, path: Path) -> None:
    """Require all backbone and both head tensors; never allow random fallback."""
    import torch

    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path), device="cpu")
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state.items()
    ):
        raise ValueError("checkpoint must be a tensor state dictionary")
    model.load_state_dict(state, strict=True)


def load_tokenizer(directory: Path) -> Any:
    """Read the complete serialized pipeline despite v5-only wrapper metadata."""
    from transformers import PreTrainedTokenizerFast

    metadata = json.loads((directory / "tokenizer_config.json").read_bytes())
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(directory / "tokenizer.json"),
        model_input_names=["input_ids", "attention_mask"],
        **{
            name: metadata[name]
            for name in (
                "bos_token",
                "eos_token",
                "cls_token",
                "sep_token",
                "pad_token",
                "unk_token",
                "mask_token",
            )
        },
    )
    # Assigning token roles must not change the serialized vocabulary or pipeline.
    from tokenizers import Tokenizer

    original = Tokenizer.from_file(str(directory / "tokenizer.json"))
    if tokenizer.get_vocab() != original.get_vocab():
        raise ValueError("tokenizer role assignment changed the vocabulary")
    # The saved backend has batch padding/truncation enabled; compare under the
    # same explicit settings as the wrapper's unpadded, untruncated probe.
    original.no_padding()
    original.no_truncation()
    probe = "Company earnings rose 10%, while debt fell."
    if tokenizer(probe)["input_ids"] != original.encode(probe).ids:
        raise ValueError("serialized tokenizer encoding differs")
    return tokenizer


# A base DeBERTa classification model initializes both its pooler and classifier.
BASE_HEAD_KEYS = frozenset(
    {
        "classifier.weight",
        "classifier.bias",
        "pooler.dense.weight",
        "pooler.dense.bias",
    }
)
LANGUAGE_HEAD_PREFIXES = ("lm_predictions.", "mask_predictions.")
