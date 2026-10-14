"""Custom pooling, exact tensor loading and development-only scope."""

from types import SimpleNamespace

import pytest

from finsight.sentiment.proprak_deberta import (
    build_model,
    load_tensor_state,
    load_tokenizer,
    masked_mean,
    validation_rows,
)


def test_validation_rejects_reserved_and_omits_multitarget() -> None:
    row = {
        "partition": "validation",
        "corpus": "sentfin",
        "native_record": {"sentiments": [{"label": "positive"}]},
    }
    mixed = {**row, "native_record": {"sentiments": [{}, {}]}}
    assert validation_rows([row, mixed]) == [row]
    with pytest.raises(ValueError, match="development"):
        validation_rows([{**row, "partition": "test"}])
    with pytest.raises(ValueError, match="nonempty"):
        validation_rows([mixed])


def test_masked_pooling_ignores_padding_but_keeps_special_tokens() -> None:
    torch = pytest.importorskip("torch")
    hidden = torch.tensor([[[2.0, 4.0], [4.0, 8.0], [999.0, 999.0]]])
    mask = torch.tensor([[1, 1, 0]])
    assert torch.equal(masked_mean(hidden, mask), torch.tensor([[3.0, 6.0]]))
    assert torch.isfinite(masked_mean(hidden, torch.zeros_like(mask))).all()


def test_serialized_pipeline_preserves_ids_and_padding(tmp_path) -> None:
    import json

    t = pytest.importorskip("transformers")
    vocabulary = tmp_path / "vocab.txt"
    vocabulary.write_text(
        "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\ncompany\nearnings\nrose\n"
    )
    reference = t.BertTokenizerFast(vocab_file=str(vocabulary))
    reference.backend_tokenizer.save(str(tmp_path / "tokenizer.json"))
    metadata = {
        name: getattr(reference, name)
        for name in (
            "bos_token",
            "eos_token",
            "cls_token",
            "sep_token",
            "pad_token",
            "unk_token",
            "mask_token",
        )
    }
    metadata["extra_special_tokens"] = ["[PAD]", "[CLS]", "[SEP]"]
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(metadata))
    loaded = load_tokenizer(tmp_path)
    texts = ["Company earnings rose", "Company"]
    expected = reference(texts, padding=True, truncation=True, max_length=8)
    actual = loaded(texts, padding=True, truncation=True, max_length=8)
    assert actual["input_ids"] == expected["input_ids"]
    assert actual["attention_mask"] == expected["attention_mask"]
    assert "token_type_ids" not in actual


@pytest.mark.filterwarnings(
    "ignore:.*torch.jit.script.*deprecated:DeprecationWarning:torch.jit._script"
)
def test_exact_loading_and_published_forward(tmp_path, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    class FakeBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(4, 2)

        def forward(self, input_ids, attention_mask):
            return SimpleNamespace(last_hidden_state=self.embedding(input_ids))

    monkeypatch.setattr(
        transformers.AutoModel, "from_config", lambda *a, **kw: FakeBackbone()
    )
    model = build_model(SimpleNamespace(hidden_size=2))
    model.eval()
    path = tmp_path / "weights.bin"
    torch.save(model.state_dict(), path)
    restored = build_model(SimpleNamespace(hidden_size=2))
    load_tensor_state(restored, path)
    restored.eval()
    ids, mask = torch.tensor([[0, 1, 2]]), torch.tensor([[1, 1, 0]])
    output = restored(ids, mask)
    pooled = model.backbone.embedding(ids)[:, :2].mean(1)
    torch.testing.assert_close(output.logits, model.cls_head(pooled))
    torch.testing.assert_close(output.regression, model.reg_head(pooled).squeeze(-1))
    assert list(restored.config.id2label.values()) == [
        "negative",
        "neutral",
        "positive",
    ]
    partial = dict(model.state_dict())
    del partial["reg_head.bias"]
    torch.save(partial, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_tensor_state(restored, path)
    torch.save({"metadata": "not a tensor"}, path)
    with pytest.raises(ValueError, match="tensor state"):
        load_tensor_state(restored, path)
