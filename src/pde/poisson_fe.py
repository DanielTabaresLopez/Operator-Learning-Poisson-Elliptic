from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass
class FEResult:
    u: np.ndarray
    converged: bool
    iters: int
    info: int


class VariableCoeffPoissonFE:
    """
    P1 finite elements on a structured triangulation of the unit square.

    PDE:
        -div(a grad u) = f  in (0,1)^2
        u = 0               on boundary

    We keep the same external interface as the FD solver:
        solve(a, f, method="cg", tol=..., maxiter=...)
    where a and f are given on an (n x n) regular grid.

    Internally:
      - vertices are the same regular-grid points
      - each square cell is split into 2 triangles
      - a and f are interpolated by nodal values
      - output u is returned on the same (n x n) grid
    """

    def __init__(self, n: int):
        if n < 2:
            raise ValueError("n must be >= 2")
        self.n = n
        self.h = 1.0 / (n - 1)

        self.xy = self._build_vertices()             # (Nv, 2)
        self.tris = self._build_triangles()          # (Nt, 3)
        self.boundary_mask = self._build_boundary_mask()
        self.interior = np.where(~self.boundary_mask)[0]
        self.boundary = np.where(self.boundary_mask)[0]

        # precompute per-triangle geometry
        self._tri_area = np.zeros(len(self.tris), dtype=np.float64)
        self._tri_grads = np.zeros((len(self.tris), 3, 2), dtype=np.float64)

        for t, tri in enumerate(self.tris):
            pts = self.xy[tri]  # (3,2)
            area, grads = self._triangle_geometry(pts)
            self._tri_area[t] = area
            self._tri_grads[t] = grads

    def _build_vertices(self) -> np.ndarray:
        xs = np.linspace(0.0, 1.0, self.n)
        ys = np.linspace(0.0, 1.0, self.n)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        return np.stack([X.reshape(-1), Y.reshape(-1)], axis=1)

    def _idx(self, i: int, j: int) -> int:
        return i * self.n + j

    def _build_triangles(self) -> np.ndarray:
        tris = []
        for i in range(self.n - 1):
            for j in range(self.n - 1):
                v00 = self._idx(i, j)
                v10 = self._idx(i + 1, j)
                v01 = self._idx(i, j + 1)
                v11 = self._idx(i + 1, j + 1)

                # split each square into two triangles
                tris.append([v00, v10, v11])
                tris.append([v00, v11, v01])
        return np.asarray(tris, dtype=np.int64)

    def _build_boundary_mask(self) -> np.ndarray:
        mask = np.zeros(self.n * self.n, dtype=bool)
        for i in range(self.n):
            for j in range(self.n):
                if i == 0 or i == self.n - 1 or j == 0 or j == self.n - 1:
                    mask[self._idx(i, j)] = True
        return mask

    @staticmethod
    def _triangle_geometry(pts: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        pts: (3,2) = [[x1,y1],[x2,y2],[x3,y3]]
        returns:
          area
          grads of barycentric basis functions: (3,2)
        """
        x1, y1 = pts[0]
        x2, y2 = pts[1]
        x3, y3 = pts[2]

        detB = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        area = 0.5 * abs(detB)
        if area <= 0.0:
            raise ValueError("Degenerate triangle encountered")

        # Gradients of P1 basis functions on a triangle
        grads = np.array([
            [y2 - y3, x3 - x2],
            [y3 - y1, x1 - x3],
            [y1 - y2, x2 - x1],
        ], dtype=np.float64) / detB

        return area, grads

    def _assemble_stiffness(self, a_vec: np.ndarray) -> sp.csr_matrix:
        Nv = self.n * self.n
        rows = []
        cols = []
        vals = []

        for t, tri in enumerate(self.tris):
            area = self._tri_area[t]
            grads = self._tri_grads[t]  # (3,2)

            # elementwise average coefficient
            a_T = float(np.mean(a_vec[tri]))

            Ke = np.zeros((3, 3), dtype=np.float64)
            for i in range(3):
                for j in range(3):
                    Ke[i, j] = a_T * area * np.dot(grads[i], grads[j])

            for i_local in range(3):
                I = tri[i_local]
                for j_local in range(3):
                    J = tri[j_local]
                    rows.append(I)
                    cols.append(J)
                    vals.append(Ke[i_local, j_local])

        K = sp.coo_matrix((vals, (rows, cols)), shape=(Nv, Nv))
        return K.tocsr()

    def _assemble_load(self, f_vec: np.ndarray) -> np.ndarray:
        """
        Lumped/simple P1-consistent load using centroid/vertex averaging:
        Fe ≈ area * f_T / 3 * [1,1,1]
        """
        Nv = self.n * self.n
        F = np.zeros(Nv, dtype=np.float64)

        for t, tri in enumerate(self.tris):
            area = self._tri_area[t]
            f_T = float(np.mean(f_vec[tri]))
            Fe = area * f_T / 3.0
            F[tri] += Fe

        return F

    def solve(
        self,
        a: np.ndarray,
        f: np.ndarray,
        method: str = "cg",
        tol: float = 1e-8,
        maxiter: int = 3000,
    ) -> FEResult:
        if a.shape != (self.n, self.n):
            raise ValueError(f"a must have shape {(self.n, self.n)}, got {a.shape}")
        if f.shape != (self.n, self.n):
            raise ValueError(f"f must have shape {(self.n, self.n)}, got {f.shape}")

        a_vec = a.reshape(-1).astype(np.float64)
        f_vec = f.reshape(-1).astype(np.float64)

        K = self._assemble_stiffness(a_vec)
        F = self._assemble_load(f_vec)

        I = self.interior
        Kii = K[I][:, I]
        Fi = F[I]  # boundary values are zero, so no correction term

        iters = 0

        def _cb(_x):
            nonlocal iters
            iters += 1

        if method == "cg":
            ui, info = spla.cg(Kii, Fi, rtol=tol, atol=0.0, maxiter=maxiter, callback=_cb)
            converged = (info == 0)
        elif method == "spsolve":
            ui = spla.spsolve(Kii, Fi)
            info = 0
            converged = True
            iters = 1
        else:
            raise ValueError("method must be 'cg' or 'spsolve'")

        u = np.zeros(self.n * self.n, dtype=np.float64)
        u[I] = ui
        u = u.reshape(self.n, self.n)

        return FEResult(
            u=u,
            converged=bool(converged),
            iters=int(iters),
            info=int(info),
        )