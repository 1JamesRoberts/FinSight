"""Compatibility import; use finsight.sentiment.models.saved_inference."""

import sys
from importlib import import_module

_module = import_module("finsight.sentiment.models.saved_inference")

sys.modules[__name__] = _module
