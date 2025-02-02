__all__ = ["__version__", "function", "manifolds", "optimizers", "Problem"]

import os

from nc.optimisers.pymanopt.gpu import function, manifolds, optimizers
from nc.optimisers.pymanopt.gpu.core.problem import Problem

from nc.optimisers.pymanopt.gpu._version import __version__


os.environ["TF_CPP_MIN_LOG_LEVEL"] = os.getenv("TF_CPP_MIN_LOG_LEVEL", "2")
