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
# ------------- DEMOGRAPHICS -------------
with tabs[0]:
    age = st.number_input("AGE", min_value=0, max_value=100, value=50)
    gender = st.selectbox("Gender", ["Male", "Female"])
    socioeconomic = st.selectbox("Socioeconomic Status", ["Low", "Middle", "High"])

# ------------- SYMPTOMS -------------
with tabs[1]:
    chest_pain = st.selectbox("Chest Pain Character", ["Typical", "Atypical", "None"])
    dyspnea = st.checkbox("Dyspnea", value=False)
    fatigue = st.checkbox("Fatigue", value=False)
    nausea = st.checkbox("Nausea", value=False)

# ------------- LAB RESULTS -------------
with tabs[2]:
    trop = st.text_input("Troponin (ng/mL)", placeholder="Enter value or leave blank")
    crp = st.text_input("CRP (mg/L)", placeholder="Enter value or leave blank")
    bnp = st.text_input("BNP", placeholder="Enter value or leave blank")
    hdl = st.text_input("HDL", placeholder="Enter value or leave blank")
    ldl = st.text_input("LDL", placeholder="Enter value or leave blank")

    # convert to float if not empty
    def parse_float(x):
        try:
            return float(x) if x.strip() != "" else np.nan
        except:
            return np.nan

    trop = parse_float(trop)
    crp = parse_float(crp)
    bnp = parse_float(bnp)
    hdl = parse_float(hdl)
    ldl = parse_float(ldl)

# ------------- ECG -------------
with tabs[3]:
    st_depression = st.checkbox("ECG ST Depression", value=False)
    t_neg = st.checkbox("T-wave Negativity", value=False)
    q_wave = st.checkbox("Q Waves", value=False)

# ------------- IMAGING -------------
with tabs[4]:
    ef = st.text_input("Ejection Fraction (%)", placeholder="Enter value or leave blank")
    mri_t2 = st.text_input("MRI_T2 (signal intensity)", placeholder="Enter value or leave blank")
    mri_lge = st.text_input("MRI_LGE (fibrosis %)", placeholder="Enter value or leave blank")

    ef = parse_float(ef)
    mri_t2 = parse_float(mri_t2)
    mri_lge = parse_float(mri_lge)

# ------------- RISK FACTORS -------------
with tabs[5]:
    dm = st.checkbox("Diabetes Mellitus (DM)", value=False)
    ht = st.checkbox("Hypertension (HT)", value=False)
    hl = st.checkbox("Hyperlipidemia (HL)", value=False)
    smoker = st.checkbox("Smoking History", value=False)


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

    missing_fields = [
        k for k, v in new_patient.items()
        if k != "CONFIRMED DIAGNOSIS"  # ignore label
        and (v is None or (isinstance(v, (int, float)) and np.isnan(v)))
        ]


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
        elif fill_choice == "Yes, fill with mean values":
            numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
            for f in missing_fields:
                if f in numeric_cols:
                    new_patient[f] = df[f].mean()
            st.session_state.new_patient = new_patient
            st.session_state.filled = True
            st.success("✅ Missing values have been filled with column means.")
            st.experimental_rerun()   # rerun once after filling
        st.stop()

    # if already filled or no missing values
    if st.session_state.filled:
        new_patient = st.session_state.new_patient

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




