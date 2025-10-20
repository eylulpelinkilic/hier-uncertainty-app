# app.py
import streamlit as st
import matplotlib.pyplot as plt
from hier_uncertainty import run_pipeline, add_patient, load_builtin_data, entropy
from scipy.stats import gaussian_kde
import numpy as np
from hier_uncertainty import plot_diagnostic_landscape
from scipy.stats import zscore
import pandas as pd

#not to let restart tsne each time
@st.cache_data
def compute_tsne(X_train_std, perplexity):
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    return tsne.fit_transform(X_train_std)

st.set_page_config(page_title="Hierarchical Uncertainty Visualization", layout="wide")
st.title("Hierarchical Uncertainty Visualization (Demo)")

# Sidebar parameters
st.sidebar.header("Pipeline Settings")
bins = st.sidebar.slider("Number of bins", 5, 50, 20)
lam = st.sidebar.slider("Lambda (within-class penalty)", 0.0, 1.0, 0.5, 0.1)
eps = st.sidebar.number_input("Epsilon (stability)", value=1e-6, format="%.1e")
perp = st.sidebar.slider("t-SNE perplexity", 5, 100, 50)
params = {"bins": bins, "lambda": lam, "eps": eps, "perplexity": perp}

# Built-in data
st.sidebar.success("Using built-in demo data.")
df = load_builtin_data()
# Initial pipeline run to compute t-SNE embedding for base visualization
X_train_emb, meta, df, num_cols = run_pipeline(
    df, label_col="CONFIRMED DIAGNOSIS", params=params
)

st.dataframe(df.head())

# Run pipeline
X_emb, meta, df, num_cols = run_pipeline(df, label_col="CONFIRMED DIAGNOSIS", params=params)

st.subheader("Three-Class Diagnostic Landscape (t-SNE + KDE)")
fig = plot_diagnostic_landscape(df, X_emb, label_col="CONFIRMED DIAGNOSIS")
fig = plot_diagnostic_landscape(df, X_train_emb, label_col="CONFIRMED DIAGNOSIS")
ax = fig.gca()
st.pyplot(fig)


# ------------------ Add new patient (Modular, Eq.(8) Integration) ------------------
st.subheader("🧍 Add New Patient (Modular Data Entry)")

tabs = st.tabs(["Demographics", "Symptoms", "Lab Results", "ECG", "Imaging", "Risk Factors"])

# ------------- DEMOGRAPHICS -------------
with tabs[0]:
    age = st.number_input("AGE", 0, 100, 50)
    gender = st.selectbox("Gender", ["Male", "Female"])
    socioeconomic = st.selectbox("Socioeconomic Status", ["Low", "Middle", "High"])

# ------------- SYMPTOMS -------------
with tabs[1]:
    chest_pain = st.selectbox("Chest Pain Character", ["Typical", "Atypical", "None"])
    dyspnea = st.checkbox("Dyspnea")
    fatigue = st.checkbox("Fatigue")
    nausea = st.checkbox("Nausea")

# ------------- LAB RESULTS -------------
with tabs[2]:
    trop = st.number_input("Troponin (ng/mL)", 0.0, 10.0, 0.3)
    crp = st.number_input("CRP (mg/L)", 0.0, 100.0, 10.0)
    bnp = st.number_input("BNP", 0.0, 2000.0, 150.0)
    hdl = st.number_input("HDL", 0.0, 100.0, 50.0)
    ldl = st.number_input("LDL", 0.0, 250.0, 120.0)

# ------------- ECG -------------
with tabs[3]:
    st_depression = st.checkbox("ECG ST Depression")
    t_neg = st.checkbox("T-wave Negativity")
    q_wave = st.checkbox("Q Waves")

# ------------- IMAGING -------------
with tabs[4]:
    ef = st.number_input("Ejection Fraction (%)", 10.0, 80.0, 55.0)
    mri_t2 = st.number_input("MRI_T2 (signal intensity)", 0.0, 10.0, 2.0)
    mri_lge = st.number_input("MRI_LGE (fibrosis %)", 0.0, 100.0, 15.0)

# ------------- RISK FACTORS -------------
with tabs[5]:
    dm = st.checkbox("Diabetes Mellitus (DM)")
    ht = st.checkbox("Hypertension (HT)")
    hl = st.checkbox("Hyperlipidemia (HL)")
    smoker = st.checkbox("Smoking History")

# ------------- SUBMIT -------------
submitted = st.button("💡 Compute Uncertainty and Visualize")

if submitted:
    # Build new patient dict (all required)
    new_patient = {
        "AGE": age,
        "TROPONIN": trop,
        "CRP": crp,
        "BNP": bnp,
        "HDL": hdl,
        "LDL": ldl,
        "EF": ef,
        "MRI_T2": mri_t2,
        "MRI_LGE": mri_lge,
        "DM": int(dm),
        "HT": int(ht),
        "HL": int(hl),
        "SMOKER": int(smoker),
        "Dyspnea": int(dyspnea),
        "Fatigue": int(fatigue),
        "Nausea": int(nausea),
        "ECG_ST_depression": int(st_depression),
        "ECG_T_neg": int(t_neg),
        "ECG_Q_waves": int(q_wave),
        "CONFIRMED DIAGNOSIS": np.nan  
    }

    # Validation: none should be None or blank
    missing_fields = [k for k, v in new_patient.items()
                      if v is None or (isinstance(v, (int, float)) and np.isnan(v))]

    # to validate if user wants to fill missing values by mean
    if missing_fields:
        st.warning(f"⚠️ Some values are missing: {', '.join(missing_fields)}")
        fill_choice = st.radio(
            "Do you want to fill missing values with the feature mean?",
            ("No, stop the process", "Yes, fill with mean values"),
            horizontal=True
        )

        if fill_choice == "No, stop the process":
            st.stop()
        else:
            for f in missing_fields:
                new_patient[f] = df[f].mean()
            st.info("Missing values have been filled with mean values.")


    # Run uncertainty pipeline on current dataset
    from hier_uncertainty import compute_uncertainty_matrix

    # compute_uncertainty_matrix returns (uncertainty_matrix, meta, num_cols)
    X_train, meta_train, num_cols = compute_uncertainty_matrix(df, label_col="CONFIRMED DIAGNOSIS", lam=params["lambda"], eps=params["eps"], bins=params["bins"])
    df_train = df

    # Fill missing features with class means
    for f in num_cols:
        if f not in new_patient:
            new_patient[f] = df_train[f].mean()

    # -------------- Eq.(8) uncertainty computation --------------
    st.info("Computing uncertainty using Eq.(8):  x_pf = H_f * z_pf / (D_JS + ε)")

    xp_f = []
    for f in num_cols:
        z_pf = (new_patient[f] - df_train[f].mean()) / (df_train[f].std() + 1e-9)
        H_f = entropy(np.histogram(df_train[f].values, bins=20)[0])
        D_JS = meta_train[f]["D_top"] if f in meta_train else 1e-9
        xp_val = H_f * z_pf / (D_JS + 1e-6)
        xp_f.append(xp_val)

    xp_f = np.array(xp_f).reshape(1, -1)

    # --- standardize and project ---
    X_train_std = (X_train - np.nanmean(X_train, axis=0)) / (np.nanstd(X_train, axis=0) + 1e-9)
    xp_f_std = (xp_f - np.nanmean(X_train, axis=0)) / (np.nanstd(X_train, axis=0) + 1e-9)

    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1).fit(X_train_std)
    _, idx = nn.kneighbors(xp_f_std)

    # --- compute t-SNE embedding for visualization ---
    from sklearn.manifold import TSNE

    
    X_train_emb = compute_tsne(X_train_std, params["perplexity"])
    new_pt_emb = X_train_emb[idx.flatten()]


    # --- visualization (Colab-style diagnostic landscape) ---
    fig = plot_diagnostic_landscape(df_train, X_train_emb, label_col="CONFIRMED DIAGNOSIS")
    ax = plt.gca()
    # Eğer grafik üzerinde zaten bir "New Patient" varsa, tekrar çizme
    existing_labels = [t.get_text() for t in ax.texts] + [h.get_label() for h in ax.legend_.texts] if ax.legend_ else []
    if "New Patient (Eq.8)" not in existing_labels:
        ax.scatter(new_pt_emb[0,0], new_pt_emb[0,1],
                s=220, color="black", marker="*", edgecolor="white", linewidth=1.2,
                label="New Patient (Eq.8)")
    ax.legend()
    st.pyplot(fig)

    ax.scatter(new_pt_emb[0,0], new_pt_emb[0,1],
           s=220, color="black", marker="*", edgecolor="white", linewidth=1.2,
           label="New Patient (Eq.8)")

    st.success("✅ New patient added and projected into diagnostic landscape.")
    st.caption("Feature-level uncertainty computed per Eq.(8) — entropy * atypicality * divergence weighting.")




