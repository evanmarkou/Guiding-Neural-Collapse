__all__ = ["autograd", "jax", "numpy", "pytorch", "tensorflow"]

from nc.optimisers.pymanopt.gpu.autodiff import backend_decorator_factory
from nc.optimisers.pymanopt.gpu.autodiff.backends._autograd import AutogradBackend
from nc.optimisers.pymanopt.gpu.autodiff.backends._jax import JaxBackend
from nc.optimisers.pymanopt.gpu.autodiff.backends._numpy import NumPyBackend
from nc.optimisers.pymanopt.gpu.autodiff.backends._pytorch import PyTorchBackend
from nc.optimisers.pymanopt.gpu.autodiff.backends._tensorflow import TensorFlowBackend


autograd = backend_decorator_factory(AutogradBackend)
jax = backend_decorator_factory(JaxBackend)
numpy = backend_decorator_factory(NumPyBackend)
pytorch = backend_decorator_factory(PyTorchBackend)
tensorflow = backend_decorator_factory(TensorFlowBackend)
