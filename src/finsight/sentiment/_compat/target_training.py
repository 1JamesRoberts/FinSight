"""Compatibility import; use finsight.sentiment.training.target."""

import sys
from importlib import import_module

_module = import_module("finsight.sentiment.training.target")

sys.modules[__name__] = _module
