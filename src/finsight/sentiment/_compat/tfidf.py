"""Compatibility import; use finsight.sentiment.models.tfidf."""

import sys
from importlib import import_module

_module = import_module("finsight.sentiment.models.tfidf")

sys.modules[__name__] = _module
