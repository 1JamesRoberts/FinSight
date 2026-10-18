"""Compatibility import; use finsight.sentiment.models.svm."""

import sys
from importlib import import_module

_module = import_module("finsight.sentiment.models.svm")

sys.modules[__name__] = _module
