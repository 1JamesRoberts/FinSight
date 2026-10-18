"""Compatibility import; use finsight.sentiment.experiments.finbert_validation."""

import sys
from importlib import import_module

_module = import_module("finsight.sentiment.experiments.finbert_validation")

sys.modules[__name__] = _module
