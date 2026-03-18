from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from dataclasses import dataclass


def harmonic_mean(a: float, b: float, eps: float = 1e-12) -> float:
    return 2.0 * a * b / (a + b + eps)


@dataclass
class PoissonFDResult:
    u: np.ndarray           # full grid (n,n), including boundary
    iters: int
    converged: bool


class VariableCoeffPoissonFD:
    """
    Solve -div(a grad u) = f on (0,1)^2 with homogeneous Dirichlet boundary conditions,
    using a 5-point FD stencil in divergence form with harmonic averaging.
    """

    def __init__(self, n: int):
        if n < 5:
            raise ValueError("n must be at least 5 for a meaningful 2D grid.")
        self.n = n
        self.h = 1.0 / (n - 1)

    def _idx(self, i: int, j: int) -> int:
        # map interior (i,j) with 1<=i,j<=n-2 to [0, (n-2)^2 - 1]
        n_int = self.n - 2
        return (i - 1) * n_int + (j - 1)

    def assemble(self, a: np.ndarray) -> sp.csr_matrix:
        """
        Assemble SPD matrix for interior unknowns.
        a must be shape (n,n), positive.
        """
        n = self.n
        h2 = self.h * self.h
        n_int = n - 2
        N = n_int * n_int

        rows, cols, vals = [], [], []

        for i in range(1, n - 1):
            for j in range(1, n - 1):
                p = self._idx(i, j)
                a_c = a[i, j]

                # edge coefficients (harmonic mean)
                a_e = harmonic_mean(a_c, a[i + 1, j])
                a_w = harmonic_mean(a_c, a[i - 1, j])
                a_n = harmonic_mean(a_c, a[i, j + 1])
                a_s = harmonic_mean(a_c, a[i, j - 1])

                diag = (a_e + a_w + a_n + a_s) / h2
                rows.append(p); cols.append(p); vals.append(diag)

                # East neighbor
                if i + 1 <= n - 2:
                    q = self._idx(i + 1, j)
                    rows.append(p); cols.append(q); vals.append(-a_e / h2)
                # West neighbor
                if i - 1 >= 1:
                    q = self._idx(i - 1, j)
                    rows.append(p); cols.append(q); vals.append(-a_w / h2)
                # North neighbor
                if j + 1 <= n - 2:
                    q = self._idx(i, j + 1)
                    rows.append(p); cols.append(q); vals.append(-a_n / h2)
                # South neighbor
                if j - 1 >= 1:
                    q = self._idx(i, j - 1)
                    rows.append(p); cols.append(q); vals.append(-a_s / h2)

        A = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
        return A

    def solve(
        self,
        a: np.ndarray,
        f: np.ndarray,
        method: str = "cg",
        tol: float = 1e-8,
        maxiter: int = 2000,
    ) -> PoissonFDResult:
        """
        Solve the PDE. a and f must be shape (n,n).
        Boundary values of u are fixed to 0.
        """
        n = self.n
        n_int = n - 2
        N = n_int * n_int

        A = self.assemble(a)

        # RHS for interior nodes
        b = np.zeros(N, dtype=np.float64)
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                p = self._idx(i, j)
                b[p] = f[i, j]

        if method == "spsolve":
            U = spla.spsolve(A, b)
            iters, converged = 0, True
        else:
            it_counter = {"k": 0}
            def _cb(_):
                it_counter["k"] += 1

            U, info = spla.cg(A, b, rtol=tol, atol=0.0, maxiter=maxiter, callback=_cb)
            iters = it_counter["k"]
            converged = (info == 0)

        # reconstruct full grid u with boundary zeros
        u = np.zeros((n, n), dtype=np.float64)
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                p = self._idx(i, j)
                u[i, j] = U[p]

        return PoissonFDResult(u=u, iters=iters, converged=converged)


class HelmholtzFD:
    """
    Solve (-beta Δ + gamma I) w = g on (0,1)^2 with Dirichlet boundary w=0,
    using a standard 5-point FD discretization on the interior.
    This is used for the 'solve twice' Gaussian sampling option.
    """

    def __init__(self, n: int, beta: float, gamma: float):
        if n < 5:
            raise ValueError("n must be at least 5.")
        self.n = n
        self.beta = beta
        self.gamma = gamma
        self.h = 1.0 / (n - 1)
        self._A = self._assemble()
        self._solver = spla.factorized(self._A.tocsc())  # cached LU

    def _idx(self, i: int, j: int) -> int:
        n_int = self.n - 2
        return (i - 1) * n_int + (j - 1)

    def _assemble(self) -> sp.csr_matrix:
        n = self.n
        h2 = self.h * self.h
        n_int = n - 2
        N = n_int * n_int

        rows, cols, vals = [], [], []
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                p = self._idx(i, j)
                diag = self.gamma + 4.0 * self.beta / h2
                rows.append(p); cols.append(p); vals.append(diag)

                # neighbors
                if i + 1 <= n - 2:
                    q = self._idx(i + 1, j)
                    rows.append(p); cols.append(q); vals.append(-self.beta / h2)
                if i - 1 >= 1:
                    q = self._idx(i - 1, j)
                    rows.append(p); cols.append(q); vals.append(-self.beta / h2)
                if j + 1 <= n - 2:
                    q = self._idx(i, j + 1)
                    rows.append(p); cols.append(q); vals.append(-self.beta / h2)
                if j - 1 >= 1:
                    q = self._idx(i, j - 1)
                    rows.append(p); cols.append(q); vals.append(-self.beta / h2)

        return sp.csr_matrix((vals, (rows, cols)), shape=(N, N))

    def solve(self, g: np.ndarray) -> np.ndarray:
        """
        g shape (n,n). Returns w shape (n,n) with Dirichlet boundary w=0.
        """
        n = self.n
        n_int = n - 2
        N = n_int * n_int

        b = np.zeros(N, dtype=np.float64)
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                p = self._idx(i, j)
                b[p] = g[i, j]

        w_int = self._solver(b)

        w = np.zeros((n, n), dtype=np.float64)
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                p = self._idx(i, j)
                w[i, j] = w_int[p]
        return w