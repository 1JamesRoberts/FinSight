"""Organization proposals preserve text and never imply ticker identity."""

from dataclasses import dataclass
from pathlib import Path

import pytest

from finsight.sentiment.organizations import (
    OrganizationDetector,
    OrganizationMention,
    organization_sentiments,
    organization_spans,
)


def entity(start=0, end=5, score=0.9, group="ORG"):
    return dict(start=start, end=end, score=score, entity_group=group)


def test_offsets_use_original_surface_not_pipeline_word():
    results = [entity(), entity(6, 12, group="PER")]
    results[0]["word"] = "Te ##sla"
    assert organization_spans("Tesla joined", results) == (
        OrganizationMention("Tesla", 0, 5, 0.9),
    )


@pytest.mark.parametrize(
    "result",
    [
        entity(-1),
        entity(end=50),
        entity(end=0),
        entity(score=float("nan")),
        entity(score=1.1),
        entity(start=True),
    ],
)
def test_bad_model_spans_fail_closed(result):
    with pytest.raises(ValueError):
        organization_spans("Tesla", [result])


def test_no_threshold_is_tuned_and_overlaps_are_rejected():
    assert organization_spans("Tesla", [entity(score=0.01)])[0].score == 0.01
    with pytest.raises(ValueError, match="overlap"):
        organization_spans("Tesla", [entity(), entity(2, 5)])


class Tokenizer:
    def __call__(self, texts, **kwargs):
        assert kwargs == dict(add_special_tokens=True, truncation=False)
        return {"input_ids": [[0] * (len(t) + 2) for t in texts]}


class Pipeline:
    def __init__(self):
        self.calls = []

    def __call__(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        return [[entity()] for _ in texts]


def test_batch_checks_complete_length_before_calling_model():
    pipe = Pipeline()
    detector = OrganizationDetector(Tokenizer(), pipe, max_length=8)
    assert (
        detector.detect_many(["Tesla", "Tesla!"])
        == [(OrganizationMention("Tesla", 0, 5, 0.9),)] * 2
    )
    assert pipe.calls[0][1] == {"batch_size": 16}
    with pytest.raises(ValueError, match="complete"):
        detector.detect_many(["too long for this tokenizer"])
    assert len(pipe.calls) == 1


@pytest.mark.parametrize("texts", [[], [""], ["  "], [None]])
def test_empty_inputs_rejected(texts):
    with pytest.raises(ValueError):
        OrganizationDetector(Tokenizer(), Pipeline()).detect_many(texts)


@dataclass
class Prediction:
    label: str = "positive"


class Predictor:
    def __init__(self):
        self.calls = []

    def predict_many(self, texts, targets):
        self.calls.append((texts, targets))
        return [Prediction() for _ in targets]


def test_candidate_sentiment_deduplicates_surface_without_inventing_identity():
    text = "Tesla and Tesla met UNICEF"
    mentions = tuple(
        OrganizationMention(text[a:b], a, b, 0.9)
        for a, b in [(0, 5), (10, 15), (20, 26)]
    )
    predictor = Predictor()
    output = organization_sentiments(text, mentions, predictor)
    assert predictor.calls == [([text, text], ["Tesla", "UNICEF"])]
    assert len(output[0]["mentions"]) == 2
    assert all(o["company_identity_verified"] is False for o in output)
    assert output[1]["sentiment"] == {"label": "positive"}


def test_no_organizations_skips_sentiment_forward():
    predictor = Predictor()
    assert organization_sentiments("No firms", (), predictor) == []
    assert predictor.calls == []


def test_sentiment_bridge_rejects_changed_source_surface():
    with pytest.raises(ValueError, match="source text"):
        organization_sentiments(
            "Tesla", (OrganizationMention("Other", 0, 5, 0.9),), Predictor()
        )


def test_empty_cli_skips_sentiment_loading(monkeypatch, capsys):
    import json

    from finsight.sentiment import organizations
    from finsight.sentiment.models.target_predictor import TargetSentimentPredictor

    class EmptyDetector:
        def detect_many(self, texts):
            return [()]

    monkeypatch.setattr(OrganizationDetector, "load", lambda *a, **kw: EmptyDetector())

    def forbidden(*args, **kwargs):
        raise AssertionError("empty candidates must not load sentiment weights")

    monkeypatch.setattr(TargetSentimentPredictor, "load", forbidden)
    monkeypatch.setattr(
        "sys.argv", ["organizations", "--text", "No firms", "--with-sentiment"]
    )
    organizations.main()
    result = json.loads(capsys.readouterr().out)
    assert result["mentions"] == result["company_sentiments"] == []
    assert result["candidate_type"] == "organization_not_verified_company"


def test_loader_verifies_files_before_optional_model_import(tmp_path: Path):
    with pytest.raises((ValueError, FileNotFoundError)):
        OrganizationDetector.load(tmp_path)


@pytest.mark.parametrize("strategy", [None, "simple", "first"])
def test_loader_selects_word_aware_default_and_preserves_simple_override(
    strategy, monkeypatch, tmp_path
):
    import sys
    from types import SimpleNamespace

    from finsight.sentiment import organizations

    captured = {}
    tokenizer = SimpleNamespace(is_fast=True)
    model = SimpleNamespace(
        config=SimpleNamespace(id2label={5: "B-ORG"}),
        eval=lambda: None,
        requires_grad_=lambda value: None,
    )

    def factory(task, **kwargs):
        captured.update(kwargs)
        return Pipeline()

    fake = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: tokenizer),
        AutoModelForTokenClassification=SimpleNamespace(
            from_pretrained=lambda *a, **kw: model
        ),
        pipeline=factory,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake)
    monkeypatch.setattr(organizations, "verify_files", lambda *args: None)
    kwargs = {} if strategy is None else {"aggregation_strategy": strategy}
    detector = OrganizationDetector.load(tmp_path, **kwargs)
    assert captured["aggregation_strategy"] == (strategy or "first")
    assert detector.aggregation_strategy == (strategy or "first")


def test_loader_rejects_unreviewed_grouping_before_model_loading(tmp_path):
    with pytest.raises(ValueError, match="aggregation"):
        OrganizationDetector.load(tmp_path, aggregation_strategy="average")


@pytest.mark.parametrize("diagnostic", [False, True])
def test_cli_scores_verified_only_unless_all_candidate_diagnostic_requested(
    monkeypatch, capsys, diagnostic
):
    import json

    from finsight.sentiment import organizations
    from finsight.sentiment.models.target_predictor import TargetSentimentPredictor

    text = "Microsoft met Tesla"

    class Detector:
        def detect_many(self, texts):
            return [
                (
                    OrganizationMention("Microsoft", 0, 9, 0.9),
                    OrganizationMention("Tesla", 14, 19, 0.9),
                )
            ]

    predictor = Predictor()
    monkeypatch.setattr(OrganizationDetector, "load", lambda *a, **kw: Detector())
    monkeypatch.setattr(TargetSentimentPredictor, "load", lambda **kw: predictor)
    args = ["organizations", "--text", text, "--with-sentiment"]
    if diagnostic:
        args.append("--score-all-candidates")
    monkeypatch.setattr("sys.argv", args)
    organizations.main()
    result = json.loads(capsys.readouterr().out)
    assert [x["status"] for x in result["resolutions"]] == [
        "verified_company",
        "unresolved",
    ]
    assert predictor.calls[0][1] == (
        ["Microsoft", "Tesla"] if diagnostic else ["Microsoft"]
    )


def test_unresolved_only_cli_never_loads_sentiment(monkeypatch, capsys):
    import json

    from finsight.sentiment import organizations
    from finsight.sentiment.models.target_predictor import TargetSentimentPredictor

    class Detector:
        def detect_many(self, texts):
            return [(OrganizationMention("Tesla", 0, 5, 0.99),)]

    monkeypatch.setattr(OrganizationDetector, "load", lambda *a, **kw: Detector())

    def forbidden(**kwargs):
        raise AssertionError("unresolved names cannot trigger loading")

    monkeypatch.setattr(TargetSentimentPredictor, "load", forbidden)
    monkeypatch.setattr(
        "sys.argv", ["organizations", "--text", "Tesla", "--with-sentiment"]
    )
    organizations.main()
    assert json.loads(capsys.readouterr().out)["company_sentiments"] == []


def test_cli_scores_confirmed_unlinked_company_without_claiming_issuer(
    monkeypatch, capsys
):
    import json

    from finsight.sentiment import organizations
    from finsight.sentiment.models.target_predictor import TargetSentimentPredictor

    class Detector:
        def detect_many(self, texts):
            return [(OrganizationMention("Toyota", 0, 6, 0.99),)]

    predictor = Predictor()
    monkeypatch.setattr(OrganizationDetector, "load", lambda *a, **kw: Detector())
    monkeypatch.setattr(TargetSentimentPredictor, "load", lambda **kw: predictor)
    monkeypatch.setattr(
        "sys.argv",
        [
            "organizations",
            "--text",
            "Toyota reported higher profits.",
            "--with-sentiment",
            "--confirmed-company",
            "Toyota",
        ],
    )
    organizations.main()
    result = json.loads(capsys.readouterr().out)
    assert result["sentiment_policy"] == "company_mentions_only"
    assert predictor.calls[0][1] == ["Toyota"]
    assert predictor.calls[0][0] == ["Toyota reported higher profits."]
    resolution = result["resolutions"][0]
    assert resolution["company_type"] == "company"
    assert resolution["identity_status"] == "unresolved"
    output = result["company_sentiments"][0]
    assert output["company_id"] is output["sec_cik"] is output["primary_ticker"] is None
    assert (
        not output["aggregation_eligible"] and not output["company_identity_verified"]
    )


def test_confirmation_does_not_invent_a_mention_or_force_model_loading(
    monkeypatch, capsys
):
    import json

    from finsight.sentiment import organizations
    from finsight.sentiment.models.target_predictor import TargetSentimentPredictor

    class Detector:
        def detect_many(self, texts):
            return [()]

    monkeypatch.setattr(OrganizationDetector, "load", lambda *a, **kw: Detector())

    def forbidden(**kwargs):
        pytest.fail("A confirmed name without a detected span must not load sentiment")

    monkeypatch.setattr(TargetSentimentPredictor, "load", forbidden)
    monkeypatch.setattr(
        "sys.argv",
        [
            "organizations",
            "--text",
            "Markets opened.",
            "--with-sentiment",
            "--confirmed-company",
            "Toyota",
        ],
    )
    organizations.main()
    assert json.loads(capsys.readouterr().out)["company_sentiments"] == []


def test_alias_cli_requires_directory_before_loading_ner(monkeypatch):
    from finsight.sentiment import organizations

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid alias CLI must not load NER")

    monkeypatch.setattr(OrganizationDetector, "load", forbidden)
    monkeypatch.setattr(
        "sys.argv",
        ["organizations", "--text", "Toyota rose", "--company-aliases", "aliases.json"],
    )
    with pytest.raises(SystemExit):
        organizations.main()
