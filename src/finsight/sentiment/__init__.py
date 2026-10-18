"""Financial sentiment models, training, evaluation and explicit experiments.

New code uses the subpackages. Historical flat imports remain available through
small aliases so notebooks and trusted local pickles continue to load.
"""

from pathlib import Path

# Python searches this directory only when a legacy flat submodule is requested.
__path__.append(str(Path(__file__).with_name("_compat")))
