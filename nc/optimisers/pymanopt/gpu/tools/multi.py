import scipy.linalg
import scipy.version
import torch


# Scipy 1.9.0 added support for calling scipy.linalg.expm on stacked matrices.
scipy_expm = scipy.linalg.expm


def multitransp(A):
    """Vectorized matrix transpose.

    ``A`` is assumed to be an array containing ``M`` matrices, each of which
    has dimension ``N x P``.
    That is, ``A`` is an ``M x N x P`` array. Multitransp then returns an array
    containing the ``M`` matrix transposes of the matrices in ``A``, each of
    which will be ``P x N``.
    """
    if A.ndim == 2:
        return A.T
    return torch.transpose(A, 1, 2)



def multihconj(A):
    """Vectorized matrix conjugate transpose."""
    return torch.conj(multitransp(A))


def multisym(A):
    """Vectorized matrix symmetrization.

    Given an array ``A`` of matrices (represented as an array of shape ``(k, n,
    n)``), returns a version of ``A`` with each matrix symmetrized, i.e.,
    every matrix ``A[i]`` satisfies ``A[i] == A[i].T``.
    """
    return 0.5 * (A + multitransp(A))


def multiherm(A):
    return 0.5 * (A + multihconj(A))


def multiskew(A):
    """Vectorized matrix skew-symmetrization.

    Similar to :func:`multisym`, but returns an array where each matrix
    ``A[i]`` is skew-symmetric, i.e., the components of ``A`` satisfy ``A[i] ==
    -A[i].T``.
    """
    return 0.5 * (A - multitransp(A))


def multiskewh(A):
    return 0.5 * (A - multihconj(A))


def multieye(k, n):
    """Array of ``k`` ``n x n`` identity matrices."""
    return torch.eye(n).repeat(k, 1, 1)


def multilogm(A, *, positive_definite=False):
    """Vectorized matrix logarithm."""
    if not positive_definite:
        raise NotImplementedError("Only positive definite matrices are currently supported")

    w, v = torch.linalg.eigh(A)
    w = torch.log(w).unsqueeze(-1)
    logmA = v @ (w * multihconj(v))
    if A.is_complex():
        return logmA
    return logmA.real


def multiexpm(A, *, symmetric=False):
    """Vectorized matrix exponential."""
    if not symmetric:
        return torch.linalg.matrix_exp(A)

    w, v = torch.linalg.eigh(A)
    w = torch.exp(w).unsqueeze(-1)
    expmA = v @ (w * multihconj(v))
    if A.is_complex():
        return expmA
    return expmA.real


def multiqr(A):
    """Vectorized QR decomposition."""
    if A.dim() not in (2, 3):
        raise ValueError("Input must be a matrix or a stacked matrix")
    
    q, r = torch.linalg.qr(A)
    # Compute signs or unit-modulus phase of entries of diagonal of r.
    s = torch.diagonal(r, dim1=-2, dim2=-1).clone()
    s[s == 0] = 1
    s = s / torch.abs(s)
    
    s = s.unsqueeze(-1)
    q = q * multitransp(s)
    r = r * torch.conj(s)
    return q, r