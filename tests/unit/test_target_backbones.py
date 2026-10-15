"""Backbone adapters preserve class semantics and native paired inputs."""

from dataclasses import replace

import pytest


@pytest.mark.parametrize("model_key", ["finbert", "roberta_base"])
def test_local_source_load_keeps_named_logits_and_reproducible_initial_state(
    tmp_path, model_key
):
    torch = pytest.importorskip("torch")
    t = pytest.importorskip("transformers")
    from finsight.sentiment.finbert import file_sha256
    from finsight.sentiment.finbert_training import FineTuneConfig
    from finsight.sentiment.target_backbones import load_backbone

    torch.set_num_threads(1)
    torch.manual_seed(7)
    is_bert = model_key == "finbert"
    directory = (
        tmp_path
        / (
            "interim/finbert-checkpoint-cache"
            if is_bert
            else "interim/transformer-checkpoint-cache"
        )
        / "models--synthetic--model/snapshots/test"
    )
    directory.mkdir(parents=True)
    vocab = directory / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nalpha\ngains\n")
    tokenizer = t.BertTokenizer(vocab_file=str(vocab))
    tokenizer.save_pretrained(directory)
    common = dict(
        vocab_size=7,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
    )
    if is_bert:
        original = t.BertForSequenceClassification(
            t.BertConfig(
                **common,
                num_labels=3,
                id2label={0: "positive", 1: "negative", 2: "neutral"},
                label2id={"positive": 0, "negative": 1, "neutral": 2},
            )
        )
    else:
        original = t.RobertaForMaskedLM(t.RobertaConfig(**common))
    original.save_pretrained(directory)
    asset = {
        "model_id": "synthetic/model",
        "revision": "test",
        "model_type": "bert" if is_bert else "roberta",
        "files": {
            p.name: {"size_bytes": p.stat().st_size, "sha256": file_sha256(p)}
            for p in directory.iterdir()
            if p.is_file()
        },
    }
    config = FineTuneConfig(device="cpu")
    _, loaded_tokenizer, loaded, initial = load_backbone(
        model_key, asset, config, tmp_path
    )
    assert loaded.config.id2label == {0: "negative", 1: "neutral", 2: "positive"}
    if is_bert:
        encoded = loaded_tokenizer("alpha gains", return_tensors="pt")
        original.eval()
        loaded.eval()
        with torch.inference_mode():
            expected = original(**encoded).logits[:, [1, 2, 0]]
            actual = loaded(**encoded).logits
        assert torch.equal(actual, expected)
        assert torch.equal(
            loaded.bert.embeddings.word_embeddings.weight,
            original.bert.embeddings.word_embeddings.weight,
        )
    _, _, _, repeated = load_backbone(model_key, asset, config, tmp_path)
    assert initial["state_sha256"] == repeated["state_sha256"]
    assert initial["rng_sha256"] == repeated["rng_sha256"]
    # Checksum verification occurs before loading weights.
    vocab.write_text(vocab.read_text() + "changed\n")
    with pytest.raises(ValueError, match="checksum"):
        load_backbone(model_key, asset, config, tmp_path)


def test_roberta_pair_uses_four_special_tokens_and_preserves_target(tmp_path):
    pytest.importorskip("torch")
    t = pytest.importorskip("transformers")
    from tokenizers import ByteLevelBPETokenizer

    from finsight.data.target_pairs import TargetExample, encode_batch

    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(
        ["Alpha gains while Beta falls"],
        vocab_size=270,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
    )
    bpe.save_model(str(tmp_path))
    tokenizer = t.RobertaTokenizer(
        vocab_file=str(tmp_path / "vocab.json"),
        merges_file=str(tmp_path / "merges.txt"),
    )
    e = TargetExample(
        "e",
        "r",
        "g",
        "train",
        0,
        "Alpha",
        "Alpha gains while Beta falls",
        "positive",
        True,
        True,
        False,
        (),
    )
    assert tokenizer.num_special_tokens_to_add(pair=True) == 4
    targets = tokenizer(e.target, add_special_tokens=False)["input_ids"]
    headline = tokenizer(e.headline, add_special_tokens=False)["input_ids"]
    encoded = encode_batch(tokenizer, [e], "target_pair", max_length=len(targets) + 7)
    assert encoded["input_ids"][0] == tokenizer.build_inputs_with_special_tokens(
        targets, headline[:3]
    )
    changed = replace(e, target="Beta")
    assert encode_batch(
        tokenizer, [e], "matched_headline_only", max_length=32
    ) == encode_batch(tokenizer, [changed], "matched_headline_only", max_length=32)
