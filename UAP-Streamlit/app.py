import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="UAP ML - Genre Anime", page_icon="🎌", layout="wide")

# =========================
# PATHS
# =========================
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "mal_anime.csv"
MODELS_DIR = ROOT / "models"

st.title("🎌 UAP Pembelajaran Mesin — Klasifikasi Genre Anime (Tabular)")
st.caption("Input dari pengguna: memilih 1 anime dari dataset, lalu memilih model untuk prediksi genre.")

if not DATA_PATH.exists():
    st.error("❌ mal_anime.csv tidak ditemukan. Taruh file mal_anime.csv di folder yang sama dengan app.py.")
    st.stop()

if not MODELS_DIR.exists():
    st.error("❌ Folder models/ tidak ditemukan. Buat folder models di sebelah app.py dan taruh file model di dalamnya.")
    st.stop()

# =========================
# HELPERS
# =========================
def exists(*parts):
    return (MODELS_DIR / Path(*parts)).exists()

def to_dense(x):
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)

def safe_get_feature_names(preprocess):
    if hasattr(preprocess, "get_feature_names_out"):
        try:
            return preprocess.get_feature_names_out()
        except Exception:
            return None
    return None

def flatten_feature_cols(obj):
    # opsional (tidak dipakai sebagai penentu shape, cuma buat debug / fallback)
    if isinstance(obj, dict):
        obj = obj.get("columns", list(obj.values()))
    if isinstance(obj, np.ndarray):
        obj = obj.tolist()

    while isinstance(obj, (list, tuple, np.ndarray)) and len(obj) > 0 and isinstance(obj[0], (list, tuple, np.ndarray)):
        obj = obj[0]

    if isinstance(obj, (list, tuple, np.ndarray)):
        flat = []
        for x in obj:
            if isinstance(x, (list, tuple, np.ndarray)):
                flat.append(x[0] if len(x) else None)
            else:
                flat.append(x)
        obj = flat
    else:
        obj = [obj]

    return [str(c) for c in obj if c is not None]

def force_dim(X_in: np.ndarray, expected_dim: int) -> np.ndarray:
    """Pad/truncate ke expected_dim."""
    if X_in.ndim == 1:
        X_in = X_in.reshape(1, -1)

    cur = X_in.shape[1]
    if cur == expected_dim:
        return X_in

    if cur > expected_dim:
        return X_in[:, :expected_dim]

    pad = expected_dim - cur
    return np.hstack([X_in, np.zeros((X_in.shape[0], pad), dtype=X_in.dtype)])

# =========================
# LOAD DATASET
# =========================
@st.cache_data
def load_df(path: Path):
    return pd.read_csv(path)

df = load_df(DATA_PATH)

# cari kolom judul
name_col = None
for c in ["title", "name", "anime_title"]:
    if c in df.columns:
        name_col = c
        break
if name_col is None:
    name_col = df.columns[0]

# =========================
# CHECK MODEL FILES
# =========================
HAS_MLP = exists("preprocess.pkl") and exists("mlp_model.h5") and exists("label_encoder.pkl")
HAS_TABNET = exists("preprocess.pkl") and exists("tabnet_model.zip") and exists("label_encoder_tabnet.pkl")
HAS_FTT = (
    exists("fttransformer.pt")
    and exists("label_encoder_ft.pkl")
    and exists("num_imputer_ft.pkl")
    and exists("scaler_ft.pkl")
    and exists("cat_encoders_ft.pkl")
    and exists("ft_feature_spec.pkl")
)

# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙️ Pengaturan")

model_choice = st.sidebar.selectbox(
    "Pilih Model",
    ["MLP (Base)", "TabNet (Pretrained 1)", "FT-Transformer (Pretrained 2)"],
)

DEBUG = st.sidebar.checkbox("Tampilkan debug", value=False)

st.sidebar.markdown("### Status file model")
st.sidebar.write(f"MLP: {'✅' if HAS_MLP else '❌'}")
st.sidebar.write(f"TabNet: {'✅' if HAS_TABNET else '❌'}")
st.sidebar.write(f"FT-Transformer: {'✅' if HAS_FTT else '❌'}")

if model_choice.startswith("MLP") and not HAS_MLP:
    st.sidebar.error("File MLP belum lengkap di models/ (preprocess.pkl, label_encoder.pkl, mlp_model.h5).")
if model_choice.startswith("TabNet") and not HAS_TABNET:
    st.sidebar.error("File TabNet belum lengkap di models/ (preprocess.pkl, label_encoder_tabnet.pkl, tabnet_model.zip).")
if model_choice.startswith("FT") and not HAS_FTT:
    st.sidebar.error("File FT-Transformer belum lengkap di models/.")

# =========================
# INPUT USER: PILIH ANIME
# =========================
st.subheader("🧾 Input Data")

query = st.text_input("Cari judul anime (opsional):", "")

view_df = df
if query.strip():
    view_df = df[df[name_col].astype(str).str.contains(query, case=False, na=False)].copy()

view_df = view_df.head(5000)

if len(view_df) == 0:
    st.warning("Tidak ada judul yang cocok dengan pencarian.")
    st.stop()

selected_idx = st.selectbox(
    "Pilih 1 anime dari dataset",
    options=view_df.index.tolist(),
    format_func=lambda i: f"[{i}] {str(df.loc[i, name_col])}",
)

row = df.loc[[selected_idx]].copy()

with st.expander("Lihat data baris terpilih"):
    st.dataframe(row)

# =========================
# FEATURE SELECTION (input untuk preprocess / FTT)
# =========================
drop_cols_candidates = [
    "anime_id", "mal_id", "uid",
    "title", "name", "anime_title",
    "synopsis", "background",
    "genre", "main_genre", "genres",
]
X_row = row.drop(columns=[c for c in drop_cols_candidates if c in row.columns], errors="ignore")

# =========================
# LOADERS
# =========================
@st.cache_resource
def load_mlp_assets():
    preprocess = joblib.load(MODELS_DIR / "preprocess.pkl")
    le = joblib.load(MODELS_DIR / "label_encoder.pkl")

    import tensorflow as tf
    mlp = tf.keras.models.load_model(MODELS_DIR / "mlp_model.h5")

    # optional feature_cols (kalau ada, cuma buat debug)
    feature_cols = None
    if exists("feature_cols.pkl"):
        try:
            feature_cols = flatten_feature_cols(joblib.load(MODELS_DIR / "feature_cols.pkl"))
        except Exception:
            feature_cols = None

    return preprocess, le, mlp, feature_cols

@st.cache_resource
def load_tabnet_assets():
    from pytorch_tabnet.tab_model import TabNetClassifier

    preprocess = joblib.load(MODELS_DIR / "preprocess.pkl")
    le = joblib.load(MODELS_DIR / "label_encoder_tabnet.pkl")
    svd = joblib.load(MODELS_DIR / "svd_tabnet.pkl") if exists("svd_tabnet.pkl") else None

    tabnet = TabNetClassifier()
    tabnet.load_model(str(MODELS_DIR / "tabnet_model.zip"))
    return preprocess, svd, le, tabnet

@st.cache_resource
def load_ftt_assets():
    import torch
    import rtdl_revisiting_models as rtdl

    y_le = joblib.load(MODELS_DIR / "label_encoder_ft.pkl")
    num_imputer = joblib.load(MODELS_DIR / "num_imputer_ft.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler_ft.pkl")
    cat_encoders = joblib.load(MODELS_DIR / "cat_encoders_ft.pkl")
    use_num, use_cat, cat_cardinalities = joblib.load(MODELS_DIR / "ft_feature_spec.pkl")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = rtdl.FTTransformer(
        n_cont_features=len(use_num),
        cat_cardinalities=cat_cardinalities,
        d_block=192,
        n_blocks=3,
        attention_n_heads=8,
        attention_dropout=0.2,
        ffn_dropout=0.1,
        residual_dropout=0.0,
        ffn_d_hidden_multiplier=4,
        d_out=len(y_le.classes_),
    ).to(device)

    state = torch.load(MODELS_DIR / "fttransformer.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()

    return device, y_le, num_imputer, scaler, cat_encoders, (use_num, use_cat, cat_cardinalities), model

# =========================
# PREDICT
# =========================
st.subheader("🔮 Prediksi")

if st.button("Prediksi Genre"):
    try:
        # =========================
        # MLP (BASE)
        # =========================
        if model_choice.startswith("MLP"):
            if not HAS_MLP:
                st.error("File MLP belum lengkap di folder models/.")
                st.stop()

            preprocess, le, mlp, feature_cols = load_mlp_assets()

            Xt = preprocess.transform(X_row)
            X_in = to_dense(Xt).astype(np.float32)

            expected_dim = mlp.input_shape[-1]  # biasanya 300
            before_shape = X_in.shape
            X_in = force_dim(X_in, expected_dim)
            after_shape = X_in.shape

            if DEBUG:
                st.write(f"🔎 Debug MLP: preprocess shape={before_shape} -> input shape={after_shape}, expected={expected_dim}")
                if feature_cols is not None:
                    st.write(f"🔎 Debug MLP: feature_cols.pkl len = {len(feature_cols)}")

            probs = mlp.predict(X_in, verbose=0)[0]
            pred_idx = int(np.argmax(probs))
            pred_label = le.inverse_transform([pred_idx])[0]
            conf = float(np.max(probs))

        # =========================
        # TABNET (PRETRAINED 1)
        # =========================
        elif model_choice.startswith("TabNet"):
            if not HAS_TABNET:
                st.error("File TabNet belum lengkap di folder models/.")
                st.stop()

            preprocess, svd, le, tabnet = load_tabnet_assets()

            Xt = preprocess.transform(X_row)
            X_in = svd.transform(Xt) if svd is not None else to_dense(Xt)

            pred_idx = int(tabnet.predict(X_in)[0])
            pred_label = le.inverse_transform([pred_idx])[0]

            try:
                probas = tabnet.predict_proba(X_in)[0]
                conf = float(np.max(probas))
            except Exception:
                conf = None

        # =========================
        # FT-TRANSFORMER (PRETRAINED 2)
        # =========================
        else:
            if not HAS_FTT:
                st.error("File FT-Transformer belum lengkap di folder models/.")
                st.stop()

            import torch

            device, y_le, num_imputer, scaler, cat_encoders, spec, ftt = load_ftt_assets()
            use_num, use_cat, _ = spec

            # numeric
            X_num = row[use_num].copy()
            X_num = num_imputer.transform(X_num)
            X_num = scaler.transform(X_num)

            # categorical -> ids
            X_cat = np.zeros((len(row), len(use_cat)), dtype=np.int64)
            for j, col in enumerate(use_cat):
                enc = cat_encoders[col]
                v = row[col].astype(str).fillna("MISSING").values

                known = set(enc.classes_)
                v = np.array([x if x in known else "MISSING" for x in v], dtype=object)

                if "MISSING" not in known:
                    enc.classes_ = np.append(enc.classes_, "MISSING")

                X_cat[:, j] = enc.transform(v)

            x_num_t = torch.tensor(X_num, dtype=torch.float32).to(device)
            x_cat_t = torch.tensor(X_cat, dtype=torch.long).to(device)

            with torch.no_grad():
                logits = ftt(x_num_t, x_cat_t)
                prob = torch.softmax(logits, dim=1).cpu().numpy()[0]

            pred_idx = int(np.argmax(prob))
            pred_label = y_le.inverse_transform([pred_idx])[0]
            conf = float(np.max(prob))

        # =========================
        # OUTPUT
        # =========================
        st.success(f"✅ Prediksi genre: **{pred_label}**")
        if conf is not None:
            st.write(f"Confidence: **{conf:.4f}**")

    except Exception as e:
        st.error("Terjadi error saat prediksi (lihat detail di bawah).")
        st.exception(e)

st.caption(f"Dataset: {DATA_PATH.name} | Models: {MODELS_DIR}")
