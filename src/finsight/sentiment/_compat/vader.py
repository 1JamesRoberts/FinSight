"""Compatibility import; use finsight.sentiment.models.vader."""

import sys
from importlib import import_module

_module = import_module("finsight.sentiment.models.vader")

sys.modules[__name__] = _module
