"""Compatibility import; use finsight.sentiment.experiments.temperature_scaling."""

import sys
from importlib import import_module

_module = import_module("finsight.sentiment.experiments.temperature_scaling")

if __name__ == "__main__":
    _module.main()
else:
    sys.modules[__name__] = _module
