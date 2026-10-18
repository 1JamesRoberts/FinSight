"""Compatibility import; use finsight.sentiment.experiments.tfidf_validation."""

import sys
from importlib import import_module

_module = import_module("finsight.sentiment.experiments.tfidf_validation")

sys.modules[__name__] = _module
