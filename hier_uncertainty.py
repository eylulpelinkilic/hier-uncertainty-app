import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

# ===================== LOAD DATA =====================
def load_builtin_data():
    """
    Loads Pelin's real dataset hier-data.xlsx (replaces random demo data).
    Cleans numeric columns and fills missing values automatically.
    """
    df = pd.read_excel("hier-data.xlsx")

    # drop empty columns
    df = df.dropna(axis=1, how="all")

    # remove columns with only one unique value
    df = df.loc[:, df.nunique() > 1]

    # drop columns with >90% NaN
    df = df.dropna(axis=1, thresh=int(len(df) * 0.1))

    # fill numeric NaNs with mean
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "CONFIRMED DIAGNOSIS"]
    df[num_cols] = df[num_cols].apply(lambda c: c.fillna(c.mean()))

    return df


# ===================== UNCERTAINTY CORE =====================
def entropy(p, base=2):
    p = np.clip(p, 0, 1)
    p = p / p.sum()
    nz = p > 0
    return -np.sum(p[nz] * np.log(p[nz]) / np.log(base))

def kl(p, q, base=2):
    p = p / p.sum()
    q = q / q.sum()
    nz = (p > 0) & (q > 0)
    return np.sum(p[nz] * (np.log(p[nz]) - np.log(q[nz])) / np.log(base))

def jsd(p, q, base=2):
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return 0.5 * (kl(p, m, base) + kl(q, m, base))

def density_to_probs_with_edges(x, edges):
    hist, _ = np.histogram(x, bins=edges)
    p = hist.astype(float) + 1e-9
    return p / p.sum()

def priors(y):
    vc = pd.Series(y).value_counts()
    total = vc.sum()
    return {k: vc[k] / total for k in vc.index}

def jsd_between_within_for_feature(x, y, A1=1, A2=2, B=3, bins=20):
    x = np.asarray(x).flatten()
    mask1, mask2, mask3 = (y == A1), (y == A2), (y == B)
    x1, x2, x3 = x[mask1], x[mask2], x[mask3]
    if len(x1) < 2 or len(x2) < 2 or len(x3) < 2:
        return np.nan, np.nan
    _, edges = np.histogram(x[np.isfinite(x)], bins=bins)
    p1 = density_to_probs_with_edges(x1, edges)
    p2 = density_to_probs_with_edges(x2, edges)
    p3 = density_to_probs_with_edges(x3, edges)
    pi = priors(y)
    alpha = pi.get(A1, 0) / max(pi.get(A1, 0) + pi.get(A2, 0), 1e-12)
    M_A = alpha * p1 + (1 - alpha) * p2
    return jsd(M_A, p3), jsd(p1, p2)

def compute_uncertainty_matrix(df, label_col, lam=0.5, eps=1e-6, bins=20):
    y = df[label_col].values
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != label_col]
    X_mat = np.zeros((len(df), len(num_cols)))
    meta = {}
    for j, f in enumerate(num_cols):
        x = df[f].values
        D_top, D_in = jsd_between_within_for_feature(x, y, bins=bins)
        if np.isnan(D_top) or np.isnan(D_in):
            X_mat[:, j] = np.nan
            continue
        S_f = max(eps, D_top - lam * D_in)
        for i, xp in enumerate(x):
            X_mat[i, j] = xp / (S_f + eps)
        meta[f] = {"D_top": D_top, "D_in": D_in, "S_f": S_f}
    return X_mat, meta, num_cols


# ===================== PIPELINE =====================
def run_pipeline(df=None, label_col="CONFIRMED DIAGNOSIS", params=None):
    if df is None:
        df = load_builtin_data()
    if params is None:
        params = {"bins": 20, "lambda": 0.5, "eps": 1e-6, "perplexity": 50}

    BINS = params["bins"]
    LAMBDA = params["lambda"]
    EPS = params["eps"]
    PERPLEXITY = params["perplexity"]

    X, meta, num_cols = compute_uncertainty_matrix(df, label_col, lam=LAMBDA, eps=EPS, bins=BINS)
    X_std = (X - np.nanmean(X, axis=0)) / (np.nanstd(X, axis=0) + 1e-9)
    X_std = np.nan_to_num(X_std)

    tsne = TSNE(n_components=2, perplexity=PERPLEXITY, random_state=42)
    X_emb = tsne.fit_transform(X_std)

    return X_emb, meta, df, num_cols


# ===================== ADD NEW PATIENT =====================
def add_patient(df, new_values: dict):
    """
    Adds a new patient row and returns the updated DataFrame.
    Example:
    {"AGE": 45, "TROPONIN": 0.4, "CRP": 12, "LVEF": 60, "CONFIRMED DIAGNOSIS": 2}
    """
    new_row = pd.DataFrame([new_values])
    df_new = pd.concat([df, new_row], ignore_index=True)
    return df_new


# ===================== DIAGNOSTIC LANDSCAPE (Colab-Style) =====================
def plot_diagnostic_landscape(df, X, label_col="CONFIRMED DIAGNOSIS", title="Three-Class Diagnostic Landscape (t-SNE + KDE)"):
    """Colab-style KDE overlay visualization."""
    # --- t-SNE embedding already computed (X) ---
    df_emb = pd.DataFrame({
        "x": X[:, 0],
        "y": X[:, 1],
        "label": df[label_col].values
    })

    grid_size = 400
    xgrid = np.linspace(df_emb.x.min(), df_emb.x.max(), grid_size)
    ygrid = np.linspace(df_emb.y.min(), df_emb.y.max(), grid_size)
    Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
    positions = np.vstack([Xgrid.ravel(), Ygrid.ravel()])

    densities = {}
    quantiles = {}
    for c in [1, 2, 3]:
        pts = df_emb[df_emb.label == c][["x", "y"]].values.T
        kde = gaussian_kde(pts, bw_method="scott")
        z = np.reshape(kde(positions).T, Xgrid.shape)
        densities[c] = z
        q = np.quantile(z, 0.60)
        quantiles[c] = q

    def alpha_norm(z):
        z = np.sqrt(z / np.percentile(z, 98))
        return np.clip(z, 0, 1)

    A1 = alpha_norm(densities[1])
    A2 = alpha_norm(densities[2])
    A3 = alpha_norm(densities[3])

    R = (densities[1] >= quantiles[1]).astype(float)
    O = (densities[2] >= quantiles[2]).astype(float)
    B = (densities[3] >= quantiles[3]).astype(float)

    Rmix = np.clip(R + 0.6 * O, 0, 1)
    Amix = np.clip(A1 + A2, 0, 1) / 2

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_facecolor((0.97, 0.97, 0.97))

    ax.imshow(
        np.dstack((Rmix, 0.5 * O, B, 0.6 * Amix + 0.4 * A3)),
        extent=[xgrid.min(), xgrid.max(), ygrid.min(), ygrid.max()],
        origin="lower"
    )

    colors = {1: "#e41a1c", 2: "#ff7f00", 3: "#377eb8"}
    for c in [1, 2, 3]:
        subset = df_emb[df_emb.label == c]
        ax.scatter(subset.x, subset.y, s=25, color=colors[c],
                   alpha=0.9, edgecolor="white", linewidth=0.3,
                   label=f"Class {c}")

    ax.legend(title="Diagnosis", loc="upper right", frameon=True)
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    return fig
