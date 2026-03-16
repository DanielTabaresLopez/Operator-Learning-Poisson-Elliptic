from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Literal, Optional

from src.pde.poisson_fd import HelmholtzFD


@dataclass(frozen=True)
class GaussianRFParams:
    n: int                  # grid size including boundaries
    beta: float             # Helmholtz parameter (smoothing)
    gamma: float            # Helmholtz parameter (mass term)
    seed: int = 0


class GaussianRF2D:
    """
    Samples a smooth Gaussian random field w on an n x n grid.

    Target covariance (informally): C = L^{-2} with L = -beta Δ + gamma I.

    We support:
      - FFT-based sampling (fast; periodic assumption).
      - FD-based "solve twice" sampling (closer to elliptic operator; Dirichlet BC for w).
    """

    def __init__(self, params: GaussianRFParams):
        self.p = params
        self.rng = np.random.default_rng(params.seed)

    def sample_fft(self) -> np.ndarray:
        """
        Spectral sampling on a periodic grid:
          w_hat(k) = (beta |k|^2 + gamma)^(-1) * xi_hat(k)
        applied twice -> power -2 covariance.

        Returns: w of shape (n, n), real-valued.
        """
        n = self.p.n
        beta = self.p.beta
        gamma = self.p.gamma

        # Frequencies (scaled)
        kx = np.fft.fftfreq(n) * (2.0 * np.pi)
        ky = np.fft.fftfreq(n) * (2.0 * np.pi)
        KX, KY = np.meshgrid(kx, ky, indexing="ij")
        lam = beta * (KX**2 + KY**2) + gamma  # eigenvalues of (-beta Δ + gamma I)

        # Complex white noise in Fourier domain
        xi_hat = (self.rng.normal(size=(n, n)) + 1j * self.rng.normal(size=(n, n))) / np.sqrt(2.0)

        # Apply L^{-1} twice -> L^{-2} covariance
        w_hat = xi_hat / (lam**1.0)  # first L^{-1}
        w_hat = w_hat / (lam**1.0)  # second L^{-1}

        w = np.fft.ifft2(w_hat).real
        w -= w.mean()
        return w

    def sample_helmholtz_fd(self, bc: Literal["dirichlet"] = "dirichlet") -> np.ndarray:
        """
        FD-based sampling: solve L z = g, then L w = z, with Dirichlet boundary.
        This is more faithful to the 'solve twice' interpretation of C = L^{-2}.

        Returns w shape (n, n) with w=0 on boundary (Dirichlet).
        """
        n = self.p.n
        beta = self.p.beta
        gamma = self.p.gamma

        helmholtz = HelmholtzFD(n=n, beta=beta, gamma=gamma)  # Dirichlet by construction

        g = self.rng.normal(size=(n, n))
        z = helmholtz.solve(g)
        w = helmholtz.solve(z)
        w -= w.mean()
        return w