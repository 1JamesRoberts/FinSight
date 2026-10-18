"""Compatibility import; use finsight.sentiment.experiments.target_comparison."""

import sys
from importlib import import_module

_module = import_module("finsight.sentiment.experiments.target_comparison")

if __name__ == "__main__":
    _module.main()
else:
    sys.modules[__name__] = _module
