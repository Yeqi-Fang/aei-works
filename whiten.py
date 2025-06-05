import numpy as np
import matplotlib.pyplot as plt
import math

# ---- parameters ----
p = 4          # dimension (matches typical F-statistic example)
N = 100_000    # Monte-Carlo samples
np.random.seed(42)

# ---- build a random SPD covariance (like M) ----
A = np.random.randn(p, p)
Sigma = A @ A.T + p * np.eye(p)   # ensure positive-definite

# ---- two whitening matrices ----
# Cholesky
L = np.linalg.cholesky(Sigma)          # Sigma = L L^T
W_chol = np.linalg.inv(L)              # L^{-1}

# PCA / eigen
lam, Q = np.linalg.eigh(Sigma)         # Sigma = Q Λ Q^T
W_pca = np.diag(lam ** -0.5) @ Q.T     # Λ^{-1/2} Q^T

# ---- generate correlated samples X ~ N(0, Sigma) ----
Z = np.random.randn(N, p)              # iid standard normal
X = Z @ L.T                            # covariance = Sigma

# ---- whiten ----
Y_ch  = (W_chol @ X.T).T               # Cholesky-whitened
Y_pca = (W_pca  @ X.T).T               # PCA-whitened

# ---- check empirical covariance (should ~ I) ----
cov_ch  = (Y_ch.T  @ Y_ch)  / N
cov_pca = (Y_pca.T @ Y_pca) / N

print("‣ Frobenius norm |Cov_ch−I| =", np.linalg.norm(cov_ch - np.eye(p)))
print("‣ Frobenius norm |Cov_pca−I|=", np.linalg.norm(cov_pca - np.eye(p)))

# ---- build 2𝔽 statistics ----
s_ch  = (Y_ch  ** 2).sum(axis=1)       # ∼ χ²₄
s_pca = (Y_pca ** 2).sum(axis=1)       # ∼ χ²₄

# ---- theoretical χ²₄ pdf ----
x_grid = np.linspace(0, 30, 500)
k = p
coeff = 1.0 / (2 ** (k/2) * math.gamma(k/2))
pdf = coeff * (x_grid ** (k/2 - 1)) * np.exp(-x_grid / 2)

# ---- Figure 1 : Cholesky-whitened statistic ----
plt.figure(figsize=(6,4))
plt.hist(s_ch, bins=120, density=True, label="Cholesky hist", alpha=0.60)
plt.plot(x_grid, pdf, linewidth=2, label="χ²₄ pdf")
plt.title("2𝔽 after Cholesky whitening (N = 100 k)")
plt.xlabel("2𝔽 value")
plt.ylabel("density")
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()

# ---- Figure 2 : PCA-whitened statistic ----
plt.figure(figsize=(6,4))
plt.hist(s_pca, bins=120, density=True, label="PCA hist", alpha=0.60)
plt.plot(x_grid, pdf, linewidth=2, label="χ²₄ pdf")
plt.title("2𝔽 after PCA whitening (N = 100 k)")
plt.xlabel("2𝔽 value")
plt.ylabel("density")
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()
