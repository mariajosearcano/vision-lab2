import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from skimage.feature import local_binary_pattern


# =============================================================================
# CONSTANTS
# =============================================================================

DATA_DIR = "."
CLASSES = ["coast", "forest", "highway"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IMG_SIZE = (64, 64)

# =============================================================================
# MODEL REGISTRY
# Keys used:
#   "model"      → sklearn estimator instance
#   "supervised" → bool (False only for clustering models like K-Means)
#   "color"      → hex color for charts
# =============================================================================

def get_model_registry() -> dict:
    return {
        "Regresión Logística": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "supervised": True,
            "color": "#4CAF50",
        },
        "Árbol de Decisión": {
            "model": DecisionTreeClassifier(max_depth=10, random_state=42),
            "supervised": True,
            "color": "#8BC34A",
        },
        "Bosque Aleatorio": {
            "model": RandomForestClassifier(n_estimators=150, random_state=42),
            "supervised": True,
            "color": "#009688",
        },
        "Teorema de Bayes": {
            "model": GaussianNB(),
            "supervised": True,
            "color": "#00BCD4",
        },
        "Máquina de Vectores de Soporte": {
            "model": SVC(kernel="rbf", C=1.0, random_state=42),
            "supervised": True,
            "color": "#3F51B5",
        },
        "K-Means": {
            "model": KMeans(n_clusters=len(CLASSES), random_state=42, n_init=10),
            "supervised": False,
            "color": "#9C27B0",
        },
        "Perceptrón Multicapa": {
            "model": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=600, random_state=42),
            "supervised": True,
            "color": "#FF5722",
        },
        "K-Vecinos Más Cercanos": {
            "model": KNeighborsClassifier(n_neighbors=5),
            "supervised": True,
            "color": "#FF9800",
        },
        "Gradient Boosting": {
            "model": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
            "supervised": True,
            "color": "#E91E63",
        },
    }

# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_rgb_features(arr: np.ndarray) -> list:
    """Mean and std for each R, G, B channel → 6 features."""
    feats = []
    for ch in range(3):
        plane = arr[:, :, ch].astype(np.float32)
        feats += [np.mean(plane), np.std(plane)]
    return feats  # [μR, σR, μG, σG, μB, σB]


def extract_hsv_features(arr: np.ndarray) -> list:
    """Mean and std for each H, S, V channel → 6 features."""
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    feats = []
    for ch in range(3):
        plane = hsv[:, :, ch].astype(np.float32)
        feats += [np.mean(plane), np.std(plane)]
    return feats  # [μH, σH, μS, σS, μV, σV]


def extract_lbp_features(arr: np.ndarray, P: int = 8, R: float = 1.0, bins: int = 10) -> list:
    """
    Local Binary Pattern texture histogram → `bins` features.
    LBP captures micro-texture information (edges, spots, flat regions).
    """
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, P, R, method="uniform")
    n_bins = P + 2  # uniform LBP has P+2 distinct patterns
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist.tolist()


METHOD_NAMES = {
    "RGB":  "RGB (Rojo · Verde · Azul)",
    "HSV":  "HSV (Matiz · Saturación · Valor)",
    "LBP":  "LBP (Patrón Binario Local)",
}

EXTRACTOR_MAP = {
    "RGB": extract_rgb_features,
    "HSV": extract_hsv_features,
    "LBP": extract_lbp_features,
}

def get_feature_vector(arr: np.ndarray, method: str, combine_rgb: bool) -> list:
    """
    Build the feature vector for one image.
    If combine_rgb=True and method != 'RGB', prepend RGB features.
    """
    primary = EXTRACTOR_MAP[method](arr)
    if combine_rgb and method != "RGB":
        rgb = extract_rgb_features(arr)
        return rgb + primary
    return primary

# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data(show_spinner=False)
def load_dataset(data_dir: str, method: str, combine_rgb: bool):
    """
    Walk class folders, open each image, extract features.
    Returns X (features), y (string labels), file paths.
    """
    X, y, paths = [], [], []

    for label in CLASSES:
        folder = os.path.join(data_dir, label)
        if not os.path.isdir(folder):
            continue
        files = [
            f for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
        ]
        for fname in files:
            fpath = os.path.join(folder, fname)
            try:
                img = Image.open(fpath).convert("RGB").resize(IMG_SIZE)
                arr = np.array(img)
                feats = get_feature_vector(arr, method, combine_rgb)
                X.append(feats)
                y.append(label)
                paths.append(fpath)
            except Exception:
                continue

    return np.array(X, dtype=np.float32), np.array(y), paths

# =============================================================================
# K-MEANS CLUSTER → LABEL MAPPING
# =============================================================================

def map_clusters_to_labels(cluster_pred: np.ndarray, y_true: np.ndarray, n: int) -> np.ndarray:
    """Map each cluster index to the majority true label in that cluster."""
    mapping = {}
    for c in range(n):
        mask = cluster_pred == c
        if mask.sum() == 0:
            mapping[c] = 0
        else:
            mapping[c] = int(np.argmax(np.bincount(y_true[mask], minlength=n)))
    return np.array([mapping[c] for c in cluster_pred])

# =============================================================================
# TRAINING & EVALUATION
# =============================================================================

def train_models(X, y, test_size: float, random_seed: int) -> dict:
    """Train every model in the registry and return structured results."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    class_names = list(le.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc,
        test_size=test_size,
        random_state=random_seed,
        stratify=y_enc,
    )

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    registry = get_model_registry()
    results = {}

    for name, cfg in registry.items():
        model = cfg["model"]
        try:
            if cfg["supervised"]:
                model.fit(X_tr, y_train)
                y_pred = model.predict(X_te)
            else:
                # Unsupervised: fit train, predict test, map clusters
                model.fit(X_tr)
                raw = model.predict(X_te)
                y_pred = map_clusters_to_labels(raw, y_test, len(class_names))

            acc = accuracy_score(y_test, y_pred)
            cm  = confusion_matrix(y_test, y_pred)
            rep = classification_report(
                y_test, y_pred,
                target_names=class_names,
                output_dict=True,
                zero_division=0,
            )
            results[name] = {
                "accuracy": acc, "confusion_matrix": cm,
                "report": rep, "class_names": class_names,
                "config": cfg, "error": None,
                "n_train": len(X_train), "n_test": len(X_test),
            }
        except Exception as exc:
            results[name] = {
                "accuracy": 0, "confusion_matrix": None,
                "report": None, "class_names": class_names,
                "config": cfg, "error": str(exc),
                "n_train": len(X_train), "n_test": len(X_test),
            }

    return results

# =============================================================================
# MATPLOTLIB HELPERS  (dark background for all plots)
# =============================================================================

DARK_BG   = "#0E1117"
PANEL_BG  = "#161B22"
TEXT_CLR  = "#C9D1D9"
GRID_CLR  = "#21262D"

def _apply_dark(fig, ax_list=None):
    fig.patch.set_facecolor(DARK_BG)
    if ax_list is None:
        ax_list = fig.axes
    for ax in ax_list:
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT_CLR, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_CLR)
        ax.title.set_color(TEXT_CLR)
        ax.xaxis.label.set_color(TEXT_CLR)
        ax.yaxis.label.set_color(TEXT_CLR)

def fig_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(4, 3.6))
    _apply_dark(fig)
    # Normalize for color, show raw counts as annotations
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_n = np.where(cm.sum(1, keepdims=True) == 0, 0,
                        cm.astype(float) / cm.sum(1, keepdims=True))
    sns.heatmap(
        cm_n, annot=cm, fmt="d",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, cmap="YlGn", linewidths=0.5, linecolor=GRID_CLR,
        cbar=False, annot_kws={"size": 11, "color": "black", "weight": "bold"},
    )
    ax.set_xlabel("Predicción", fontsize=9)
    ax.set_ylabel("Etiqueta real", fontsize=9)
    plt.tight_layout()
    return fig

def fig_accuracy_bar(results: dict):
    valid = {k: v for k, v in results.items() if v["error"] is None}
    # Sort descending
    items = sorted(valid.items(), key=lambda kv: kv[1]["accuracy"], reverse=True)
    names  = [k for k, _ in items]
    accs   = [v["accuracy"] * 100 for _, v in items]
    colors = [v["config"]["color"] for _, v in items]

    short = {
        "Máquina de Vectores de Soporte": "SVM",
        "Perceptrón Multicapa": "MLP",
        "Bosque Aleatorio": "B. Aleatorio",
        "Regresión Logística": "Reg. Logística",
        "Árbol de Decisión": "Árbol Dec.",
        "Teorema de Bayes": "Bayes",
        "K-Means": "K-Means",
        "K-Vecinos Más Cercanos": "KNN",       
        "Gradient Boosting": "Grad. Boosting",
    }
    labels = [short.get(n, n) for n in names]

    fig, ax = plt.subplots(figsize=(8, 3.6))
    _apply_dark(fig)
    bars = ax.barh(labels, accs, color=colors, alpha=0.88, height=0.55, edgecolor="none")
    for bar, acc in zip(bars, accs):
        ax.text(
            min(acc + 0.8, 108), bar.get_y() + bar.get_height() / 2,
            f"{acc:.1f}%", va="center", ha="left",
            color="white", fontsize=9, fontweight="bold",
        )
    ax.set_xlim(0, 112)
    ax.set_xlabel("Precisión (%)", fontsize=9)
    ax.axvline(100, color=GRID_CLR, linestyle="--", alpha=0.6)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.grid(axis="x", color=GRID_CLR, linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig

def fig_per_class_metrics(report: dict, class_names: list):
    metrics = ["precision", "recall", "f1-score"]
    x = np.arange(len(class_names))
    w = 0.26
    mc = ["#4CAF50", "#2196F3", "#FF9800"]
    labels_es = ["Precisión", "Recall", "F1-Score"]

    fig, ax = plt.subplots(figsize=(5, 3.2))
    _apply_dark(fig)
    for i, (met, c, lbl) in enumerate(zip(metrics, mc, labels_es)):
        vals = [report.get(cn, {}).get(met, 0) for cn in class_names]
        ax.bar(x + i * w, vals, w, label=lbl, color=c, alpha=0.85)
    ax.set_xticks(x + w)
    ax.set_xticklabels(class_names, fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Valor", fontsize=8)
    ax.legend(fontsize=7, labelcolor=TEXT_CLR,
               facecolor=PANEL_BG, edgecolor=GRID_CLR, loc="upper right")
    ax.grid(axis="y", color=GRID_CLR, linestyle="--", alpha=0.5)
    plt.tight_layout()
    return fig

# =============================================================================
# STREAMLIT PAGE CONFIG & CUSTOM CSS
# =============================================================================

st.set_page_config(
    page_title="Clasificador de Imágenes ML",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

def apply_styles():
    st.markdown("""
    <style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Global ── */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: #0E1117;
        color: #C9D1D9;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #0D1117;
        border-right: 1px solid #21262D;
    }
    section[data-testid="stSidebar"] * { color: #C9D1D9 !important; }
    section[data-testid="stSidebar"] .stSelectbox > div,
    section[data-testid="stSidebar"] .stSlider > div { color: #C9D1D9; }

    /* ── Main header ── */
    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #4CAF50 0%, #00BCD4 60%, #8BC34A 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 0 0.2rem 0;
        line-height: 1.1;
    }
    .hero-sub {
        font-size: 0.95rem;
        color: #8B949E;
        margin-bottom: 1.5rem;
    }

    /* ── Section title ── */
    .section-title {
        font-family: 'Syne', sans-serif;
        font-size: 1.15rem;
        font-weight: 700;
        color: #E6EDF3;
        border-left: 3px solid #4CAF50;
        padding-left: 0.6rem;
        margin: 1.5rem 0 0.8rem 0;
    }

    /* ── Model card ── */
    .model-card {
        background: #161B22;
        border: 1px solid #21262D;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.5rem;
        transition: border-color 0.2s;
    }
    .model-card:hover { border-color: #30363D; }
    .model-card-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-family: 'Syne', sans-serif;
        font-size: 1rem;
        font-weight: 700;
        color: #E6EDF3;
        margin-bottom: 0.5rem;
    }
    .accuracy-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0.3rem 0;
    }
    .unsupervised-tag {
        font-size: 0.7rem;
        background: #21262D;
        color: #8B949E;
        border-radius: 999px;
        padding: 0.15rem 0.5rem;
        display: inline-block;
        margin-left: 0.3rem;
    }

    /* ── Stat pill ── */
    .stat-pill {
        display: inline-block;
        background: #21262D;
        border-radius: 6px;
        padding: 0.3rem 0.7rem;
        font-size: 0.82rem;
        color: #8B949E;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    .stat-pill b { color: #E6EDF3; }

    /* ── Info banner ── */
    .info-banner {
        background: #0D1117;
        border: 1px solid #21262D;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        font-size: 0.85rem;
        color: #8B949E;
        margin-bottom: 1rem;
    }

    /* ── Feature tag ── */
    .feat-tag {
        display: inline-block;
        background: #1C2526;
        border: 1px solid #2D4A2D;
        color: #4CAF50;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        border-radius: 4px;
        padding: 0.15rem 0.45rem;
        margin: 0.1rem;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: #0D1117;
        border-bottom: 1px solid #21262D;
        gap: 0.3rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #8B949E;
        font-family: 'DM Sans', sans-serif;
        font-size: 0.85rem;
        border-radius: 6px 6px 0 0;
        padding: 0.4rem 0.8rem;
    }
    .stTabs [aria-selected="true"] {
        background: #161B22 !important;
        color: #4CAF50 !important;
        border-bottom: 2px solid #4CAF50 !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        font-size: 0.95rem;
        background: linear-gradient(135deg, #2D5A2D, #1B3A3A);
        color: #4CAF50;
        border: 1px solid #2D5A2D;
        border-radius: 8px;
        padding: 0.55rem 1.5rem;
        width: 100%;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #3D7A3D, #2B5A5A);
        color: #81C784;
        border-color: #4CAF50;
        transform: translateY(-1px);
    }

    /* ── Divider ── */
    hr { border-color: #21262D; }

    /* ── Metrics ── */
    [data-testid="metric-container"] {
        background: #161B22;
        border: 1px solid #21262D;
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
    }
    </style>
    """, unsafe_allow_html=True)

# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding:1rem 0 0.5rem 0;'>
            <div style='font-family:Syne,sans-serif; font-size:1.4rem; font-weight:800;
                        background:linear-gradient(135deg,#4CAF50,#00BCD4);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                        background-clip:text;'>🌿 VisionML</div>
            <div style='font-size:0.72rem; color:#8B949E; margin-top:0.2rem;'>
                Clasificación de escenas naturales
            </div>
        </div>
        <hr style='border-color:#21262D; margin:0.5rem 0;'>
        """, unsafe_allow_html=True)

        st.markdown("**Configuración del análisis**")

        method = st.selectbox(
            "Método de análisis",
            options=["RGB", "HSV", "LBP"],
            format_func=lambda k: METHOD_NAMES[k],
            help=(
                "RGB: estadísticas de canales de color  \n"
                "HSV: matiz, saturación y brillo  \n"
                "LBP: textura mediante patrones binarios locales"
            ),
        )

        combine_rgb = False
        if method != "RGB":
            combine_rgb = st.checkbox(
                "Combinar con características RGB",
                value=False,
                help="Concatena las características RGB al vector del método seleccionado.",
            )

        st.markdown("---")
        st.markdown("**Parámetros de entrenamiento**")

        test_size = st.slider(
            "Proporción de prueba", 0.10, 0.40, 0.20, 0.05,
            help="Fracción de imágenes reservadas para evaluar (no usadas en entrenamiento).",
        )
        random_seed = st.number_input(
            "Semilla aleatoria", min_value=0, max_value=9999, value=42, step=1,
            help="Fija la aleatoriedad para reproducibilidad.",
        )

        st.markdown("---")

        run = st.button("Entrenar modelos")

    return method, combine_rgb, test_size, int(random_seed), run


# =============================================================================
# RENDER MODEL CARD
# =============================================================================

def render_model_result(name: str, res: dict):
    cfg   = res["config"]
    acc_color = cfg["color"]
    acc   = res["accuracy"]
    sup   = cfg["supervised"]

    unsup_tag = "" if sup else '<span class="unsupervised-tag">no supervisado</span>'
    acc_pct   = f"{acc * 100:.1f}%"

    # Determine color of accuracy value
    if acc >= 0.75:
        acc_color = "#4CAF50"
    elif acc >= 0.55:
        acc_color = "#FF9800"
    else:
        acc_color = "#f44336"

    st.markdown(f"""
    <div class="model-card">
        <div class="model-card-header">
            {name} {unsup_tag}
        </div>
        <div class="accuracy-badge" style="color:{acc_color}">{acc_pct}</div>
        <div style="font-size:0.78rem; color:#8B949E;">precisión global</div>
    </div>
    """, unsafe_allow_html=True)

    if res["error"]:
        st.error(f"Error al entrenar: {res['error']}")
        return

    cm          = res["confusion_matrix"]
    report      = res["report"]
    class_names = res["class_names"]

    col_cm, col_pm, col_info = st.columns([1.1, 1.2, 0.9])

    with col_cm:
        st.markdown('<div class="section-title" style="font-size:0.85rem;">Matriz de Confusión</div>',
                    unsafe_allow_html=True)
        st.pyplot(fig_confusion_matrix(cm, class_names), use_container_width=True)

    with col_pm:
        st.markdown('<div class="section-title" style="font-size:0.85rem;">Métricas por clase</div>',
                    unsafe_allow_html=True)
        st.pyplot(fig_per_class_metrics(report, class_names), use_container_width=True)

    with col_info:
        st.markdown('<div class="section-title" style="font-size:0.85rem;">Detalle</div>',
                    unsafe_allow_html=True)

        macro = report.get("macro avg", {})
        st.markdown(f"""
        <div class="stat-pill"><b>Precisión macro</b><br>{macro.get('precision', 0):.3f}</div>
        <div class="stat-pill"><b>Recall macro</b><br>{macro.get('recall', 0):.3f}</div>
        <div class="stat-pill"><b>F1 macro</b><br>{macro.get('f1-score', 0):.3f}</div>
        """, unsafe_allow_html=True)

        # Per-class support
        st.markdown('<div style="margin-top:0.6rem; font-size:0.78rem; color:#8B949E;">Soporte por clase</div>',
                    unsafe_allow_html=True)
        for cls in class_names:
            sup_val = int(report.get(cls, {}).get("support", 0))
            f1_val  = report.get(cls, {}).get("f1-score", 0)
            st.markdown(
                f'<div class="stat-pill"><b>{cls}</b> — F1: {f1_val:.2f} ({sup_val} imgs)</div>',
                unsafe_allow_html=True
            )

# =============================================================================
# MAIN APP
# =============================================================================

def main():
    apply_styles()

    method, combine_rgb, test_size, random_seed, run = render_sidebar()

    # ── Hero header ──────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-title">Clasificador de Escenas</div>
    <div class="hero-sub">
        Extracción de características · Múltiples modelos ML · Análisis comparativo
    </div>
    """, unsafe_allow_html=True)

    # ── Feature description banner ────────────────────────────────────────────
    method_desc = {
        "RGB": ("Canales: Rojo · Verde · Azul",
                ["μR", "σR", "μG", "σG", "μB", "σB"],
                6),
        "HSV": ("Canales: Matiz · Saturación · Valor",
                ["μH", "σH", "μS", "σS", "μV", "σV"],
                6),
        "LBP": ("Histograma de Patrones Binarios Locales (textura)",
                [f"bin_{i}" for i in range(10)],
                10),
    }
    m_label, m_feats, m_n = method_desc[method]
    extra = 6 if (combine_rgb and method != "RGB") else 0
    total_feats = m_n + extra

    tags_html = "".join(f'<span class="feat-tag">{f}</span>' for f in m_feats)
    if extra:
        tags_html += "".join(
            f'<span class="feat-tag" style="color:#00BCD4;">{f}</span>'
            for f in ["μR", "σR", "μG", "σG", "μB", "σB"]
        )

    st.markdown(f"""
    <div class="info-banner">
        <b style="color:#4CAF50;">{METHOD_NAMES[method]}</b>
        {'<span style="color:#00BCD4;"> + RGB combinado</span>' if extra else ''} — 
        {m_label} — 
        <b style="color:#E6EDF3;">{total_feats} características por imagen</b><br>
        <div style="margin-top:0.4rem;">{tags_html}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Train trigger ─────────────────────────────────────────────────────────
    if run:
        st.session_state["trained"]     = False
        st.session_state["results"]     = None
        st.session_state["dataset_info"] = None

        with st.spinner("Cargando imágenes y extrayendo características..."):
            X, y, paths = load_dataset(DATA_DIR, method, combine_rgb)

        if len(X) == 0:
            st.error("No se encontraron imágenes. Verifica que las carpetas "
                     "`coast`, `forest` y `highway` existen y contienen imágenes.")
            return

        # Class distribution
        unique, counts = np.unique(y, return_counts=True)
        dist = dict(zip(unique, counts))

        with st.spinner("Entrenando modelos…"):
            results = train_models(X, y, test_size, random_seed)

        st.session_state["trained"]      = True
        st.session_state["results"]      = results
        st.session_state["dataset_info"] = {
            "n_total": len(X), "n_feats": X.shape[1],
            "dist": dist, "method": method,
            "combine": combine_rgb, "test_size": test_size,
        }

    # ── Show results ──────────────────────────────────────────────────────────
    if st.session_state.get("trained") and st.session_state.get("results"):
        results  = st.session_state["results"]
        dinfo    = st.session_state["dataset_info"]

        # ── Dataset summary ──
        st.markdown('<div class="section-title">📊 Resumen del conjunto de datos</div>',
                    unsafe_allow_html=True)

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Total imágenes", dinfo["n_total"])
        with c2:
            st.metric("Características", dinfo["n_feats"])
        with c3:
            st.metric("Entrenamiento",
                      f"{int(dinfo['n_total'] * (1 - dinfo['test_size']))}")
        with c4:
            st.metric("Prueba",
                      f"{int(dinfo['n_total'] * dinfo['test_size'])}")
        with c5:
            best = max(
                (r for r in results.values() if r["error"] is None),
                key=lambda r: r["accuracy"],
                default=None,
            )
            if best:
                st.metric("Mejor precisión",
                          f"{best['accuracy'] * 100:.1f}%")

        # Class dist pills
        dist_html = "".join(
            f'<span class="stat-pill">{cls}: <b>{cnt} imgs</b></span>'
            for cls, cnt in dinfo["dist"].items()
        )
        st.markdown(f'<div style="margin:0.3rem 0 1rem 0;">{dist_html}</div>',
                    unsafe_allow_html=True)

        # ── Comparison chart ──
        st.markdown('<div class="section-title">Comparativa de precisión</div>',
                    unsafe_allow_html=True)
        st.pyplot(fig_accuracy_bar(results), use_container_width=True)

        # ── Per-model details ──
        st.markdown('<div class="section-title">Resultados por modelo</div>',
                    unsafe_allow_html=True)

        tabs = st.tabs(list(results.keys()))

        for tab, (name, res) in zip(tabs, results.items()):
            with tab:
                render_model_result(name, res)

    elif not st.session_state.get("trained"):
        # ── Welcome / instructions ────────────────────────────────────────────
        st.markdown('<div class="section-title">Instrucciones</div>',
                    unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        steps = [
            ("1️⃣", "Selecciona método", "Elige RGB, HSV o LBP en el panel lateral."),
            ("2️⃣", "Ajusta parámetros", "Configura la proporción de prueba y la semilla."),
            ("3️⃣", "Entrena modelos", "Pulsa **Entrenar modelos** para iniciar."),
        ]
        for col, (num, title, desc) in zip([c1, c2, c3], steps):
            with col:
                st.markdown(f"""
                <div class="model-card" style="text-align:center; padding:1.5rem 1rem;">
                    <div style="font-size:2rem;">{num}</div>
                    <div style="font-family:Syne,sans-serif; font-weight:700;
                                color:#E6EDF3; margin:0.4rem 0;">{title}</div>
                    <div style="font-size:0.83rem; color:#8B949E;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">Métodos de análisis disponibles</div>',
                    unsafe_allow_html=True)
        ma, mb, mc = st.columns(3)
        methods_info = [
            ("RGB",
             "Estadísticas del espacio de color estándar. Captura la distribución de colores "
             "en los canales Rojo, Verde y Azul.",
             ["μR, σR", "μG, σG", "μB, σB"], "#4CAF50"),
            ("HSV",
             "Matiz, Saturación y Valor. Más robusto ante cambios de iluminación y mejor "
             "para distinguir escenas por tonalidad dominante.",
             ["μH, σH", "μS, σS", "μV, σV"], "#00BCD4"),
            ("LBP — Patrón Binario Local",
             "Características de textura. Describe la microestructura de la imagen comparando "
             "cada píxel con sus vecinos. Ideal para detectar patrones de superficie.",
             ["histograma uniforme", "P=8, R=1", "10 dimensiones"], "#9C27B0"),
        ]
        for col, (ttl, desc, feats, color) in zip([ma, mb, mc], methods_info):
            with col:
                feat_tags = "".join(f'<span class="feat-tag" style="color:{color};">{f}</span>'
                                    for f in feats)
                st.markdown(f"""
                <div class="model-card" style="height:100%;">
                    <div style="font-family:Syne,sans-serif; font-weight:700;
                                color:{color}; margin:0.4rem 0;">{ttl}</div>
                    <div style="font-size:0.82rem; color:#8B949E; line-height:1.55;
                                margin-bottom:0.6rem;">{desc}</div>
                    {feat_tags}
                </div>
                """, unsafe_allow_html=True)

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()