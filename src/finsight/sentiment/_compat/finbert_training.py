"""Compatibility import; use finsight.sentiment.training.sentence."""

import sys
from importlib import import_module

_module = import_module("finsight.sentiment.training.sentence")

sys.modules[__name__] = _module
