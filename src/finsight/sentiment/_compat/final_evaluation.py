"""Compatibility import; use finsight.sentiment.evaluation.final."""

import sys
from importlib import import_module

_module = import_module("finsight.sentiment.experiments.final_evaluation")

if __name__ == "__main__":
    _module.main()
else:
    sys.modules[__name__] = _module
