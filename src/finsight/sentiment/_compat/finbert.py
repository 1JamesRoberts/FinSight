"""Compatibility import; use finsight.sentiment.models.finbert."""

import sys
from importlib import import_module

_module = import_module("finsight.sentiment.models.finbert")

sys.modules[__name__] = _module
