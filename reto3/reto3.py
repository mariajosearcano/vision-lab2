import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score)
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap

# ── Configuración de página ───────────────────────────────────────────────────
st.set_page_config(page_title="Clasificador de Iris", page_icon="🌺", layout="wide")
st.title("Clasificador de Flores Iris")
st.markdown("Compara **K-Means** y **Perceptrón Multicapa (MLP)** sobre el conjunto de datos Iris.")

# ── Barra lateral ─────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuración")

model_choice = st.sidebar.selectbox(
    "Elige un modelo",
    ["K-Means", "Perceptrón Multicapa (MLP)"],
)
test_size    = st.sidebar.slider("Tamaño del conjunto de prueba", 0.1, 0.5, 0.2, 0.05)
random_state = st.sidebar.number_input("Semilla aleatoria", 0, 999, 42, 1)

st.sidebar.markdown("---")
st.sidebar.subheader("Hiperparámetros del modelo")

if model_choice == "K-Means":
    n_clusters  = st.sidebar.slider("Número de clústeres (k)", 2, 8, 3)
    max_iter_km = st.sidebar.slider("Iteraciones máximas", 100, 1000, 300, 50)
    n_init_km   = st.sidebar.slider("Número de inicializaciones", 5, 20, 10)
else:
    hidden_raw   = st.sidebar.text_input("Neuronas por capa oculta (ej: 64,32)", value="64,32")
    activation   = st.sidebar.selectbox("Función de activación", ["relu", "tanh", "logistic"])
    max_iter_mlp = st.sidebar.slider("Iteraciones máximas", 100, 2000, 500, 100)
    learning_rate_init = st.sidebar.select_slider(
        "Tasa de aprendizaje inicial",
        options=[0.0001, 0.001, 0.01, 0.1],
        value=0.001,
    )

# ── Cargar datos ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data(path_or_buf):
    df = pd.read_csv(path_or_buf)
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])
    return df

script_dir = Path(__file__).parent
csv_files  = list(script_dir.glob("*.csv"))
if csv_files:
    df = load_data(csv_files[0])
else:
    st.warning("No se encontró ningún CSV en la carpeta del script. Carga uno desde la barra lateral.")
    st.stop()

# ── Preparar características ──────────────────────────────────────────────────
feature_cols = [c for c in df.columns if c != "Species"]
X  = df[feature_cols].values
le = LabelEncoder()
y  = le.fit_transform(df["Species"].values)
class_names = le.classes_

scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
)

PAL_FG = ["#1a5cb5", "#b54a1a", "#1ab53c"]
PAL_BG = ["#aec6f0", "#f0c8ae", "#aef0b4"]
MKRS   = ["o", "s", "^"]

# ── Construir y ajustar modelo ────────────────────────────────────────────────
if model_choice == "K-Means":
    model = KMeans(
        n_clusters=n_clusters, max_iter=max_iter_km,
        n_init=n_init_km, random_state=random_state
    )
    model.fit(X_scaled)
    raw_labels = model.labels_

    # Asignar etiquetas de clúster a clases reales por mayoría de votos
    cluster_to_class = {}
    for cluster_id in range(n_clusters):
        mask = raw_labels == cluster_id
        if mask.sum() == 0:
            cluster_to_class[cluster_id] = 0
        else:
            cluster_to_class[cluster_id] = int(np.bincount(y[mask]).argmax())

    y_pred_all = np.array([cluster_to_class[lbl] for lbl in raw_labels])
    y_pred     = y_pred_all  # full dataset (unsupervised, no train/test split concept)
    y_true     = y
    accuracy   = accuracy_score(y_true, y_pred)
    cv_scores  = np.array([accuracy])  # K-Means has no CV; show single accuracy

    inertia     = model.inertia_
    n_iter_done = model.n_iter_

else:  # MLP
    try:
        hidden_layers = tuple(int(x.strip()) for x in hidden_raw.split(",") if x.strip())
    except ValueError:
        st.error("El formato de capas ocultas no es válido. Usa números separados por coma, ej: 64,32")
        st.stop()

    model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation=activation,
        max_iter=max_iter_mlp,
        learning_rate_init=learning_rate_init,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.1,
    )
    model.fit(X_train, y_train)
    y_pred    = model.predict(X_test)
    y_true    = y_test
    accuracy  = accuracy_score(y_true, y_pred)
    cv_scores = cross_val_score(model, X_scaled, y, cv=5)

# ── Fila de métricas principales ──────────────────────────────────────────────
st.markdown("---")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Modelo", model_choice)
c2.metric("Exactitud", f"{accuracy:.2%}")
if model_choice == "K-Means":
    c3.metric("Inercia", f"{inertia:.1f}")
    c4.metric("Iteraciones realizadas", n_iter_done)
else:
    c3.metric("Media CV (5 pliegues)", f"{cv_scores.mean():.2%}")
    c4.metric("Desv. estándar CV", f"± {cv_scores.std():.2%}")
st.markdown("---")

# ── Izquierda: Matriz de confusión | Derecha: viz específica ──────────────────
left, right = st.columns(2)

with left:
    st.subheader("Matriz de Confusión")
    fig_cm, ax_cm = plt.subplots(figsize=(4, 3.5))
    ConfusionMatrixDisplay(
        confusion_matrix(y_true, y_pred), display_labels=class_names
    ).plot(ax=ax_cm, colorbar=False, cmap="Blues")
    ax_cm.set_xticklabels(class_names, rotation=20, ha="right", fontsize=8)
    ax_cm.set_yticklabels(class_names, fontsize=8)
    ax_cm.set_xlabel("Etiqueta predicha", fontsize=9)
    ax_cm.set_ylabel("Etiqueta real", fontsize=9)
    fig_cm.tight_layout()
    st.pyplot(fig_cm)

with right:
    # ── K-Means: gráfico de codo + PCA clusters ───────────────────────────────
    if model_choice == "K-Means":
        st.subheader("Análisis de Clústeres (PCA 2D)")

        # PCA para visualización
        pca    = PCA(n_components=2, random_state=random_state)
        X_pca  = pca.fit_transform(X_scaled)
        centers_pca = pca.transform(model.cluster_centers_)

        fig_cl, ax_cl = plt.subplots(figsize=(5.5, 4))
        for cls in range(len(class_names)):
            mask = y == cls
            ax_cl.scatter(
                X_pca[mask, 0], X_pca[mask, 1],
                c=PAL_FG[cls], marker=MKRS[cls],
                edgecolors="white", linewidths=0.5, s=55,
                label=f"Real: {class_names[cls]}", zorder=3
            )
        # Dibujar centroides
        ax_cl.scatter(
            centers_pca[:, 0], centers_pca[:, 1],
            c="black", marker="X", s=160, zorder=5,
            edgecolors="white", linewidths=0.8, label="Centroides"
        )
        # Colorear fondo por clúster asignado
        h = 0.05
        x_min, x_max = X_pca[:, 0].min()-0.8, X_pca[:, 0].max()+0.8
        y_min, y_max = X_pca[:, 1].min()-0.8, X_pca[:, 1].max()+0.8
        xx_g, yy_g = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        grid_pca   = np.c_[xx_g.ravel(), yy_g.ravel()]
        grid_orig  = pca.inverse_transform(grid_pca)
        Z_raw      = model.predict(grid_orig)
        Z_cls      = np.array([cluster_to_class[z] for z in Z_raw]).reshape(xx_g.shape)
        ax_cl.contourf(xx_g, yy_g, Z_cls, alpha=0.20, cmap=ListedColormap(PAL_BG))

        exp_var = pca.explained_variance_ratio_ * 100
        ax_cl.set_xlabel(f"PC1 ({exp_var[0]:.1f}% varianza)", fontsize=9)
        ax_cl.set_ylabel(f"PC2 ({exp_var[1]:.1f}% varianza)", fontsize=9)
        ax_cl.set_title("Clústeres K-Means proyectados con PCA", fontsize=10)
        ax_cl.legend(fontsize=7.5, loc="upper right")
        fig_cl.tight_layout()
        st.pyplot(fig_cl)

    # ── MLP: curva de pérdida ─────────────────────────────────────────────────
    else:
        st.subheader("Función de activación — primeras neuronas")
        
        x_vals = np.linspace(-10, 10, 300)
        fig_act, ax_act = plt.subplots(figsize=(5.5, 4))
        
        # grab weights and biases from first layer
        W0 = model.coefs_[0]      # shape: (n_features, n_neurons_layer1)
        b0 = model.intercepts_[0] # shape: (n_neurons_layer1,)
        
        for i in range(min(3, W0.shape[1])):
            w = W0[:, i].mean()   # average weight for neuron i
            b = b0[i]
            z = w * x_vals + b
            
            if activation == "logistic":
                y_vals = 1 / (1 + np.exp(-z))
            elif activation == "tanh":
                y_vals = np.tanh(z)
            else:  # relu
                y_vals = np.maximum(0, z)
            
            ax_act.plot(x_vals, y_vals, label=f"neurona {i+1} | w={w:.2f}, b={b:.2f}")
        
        ax_act.set_xlabel("x", fontsize=9)
        ax_act.set_ylabel("f(x)", fontsize=9)
        ax_act.set_title(f"Activación ({activation}) — capa 1", fontsize=10)
        ax_act.legend(fontsize=7.5)
        fig_act.tight_layout()
        st.pyplot(fig_act)

# ── Reporte de clasificación ──────────────────────────────────────────────────
st.markdown("---")
st.subheader("Reporte de Clasificación")
report    = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
report_df = pd.DataFrame(report).T.round(3)
st.dataframe(report_df.style.background_gradient(
    cmap="Blues", subset=["precision", "recall", "f1-score"]))

# ── Frontera de decisión 2D ───────────────────────────────────────────────────
st.markdown("---")
st.subheader("Frontera de Decisión (dos características)")

feat_x = st.selectbox("Eje X", feature_cols, index=0)
feat_y = st.selectbox("Eje Y", feature_cols, index=2)
idx_x, idx_y = feature_cols.index(feat_x), feature_cols.index(feat_y)

X2       = X_scaled[:, [idx_x, idx_y]]
X2_train, _, y2_train, _ = train_test_split(
    X2, y, test_size=test_size, random_state=random_state, stratify=y
)

if model_choice == "K-Means":
    m2 = KMeans(n_clusters=n_clusters, max_iter=max_iter_km,
                n_init=n_init_km, random_state=random_state)
    m2.fit(X2)
    raw2 = m2.labels_
    c2c  = {}
    for cid in range(n_clusters):
        mask = raw2 == cid
        c2c[cid] = int(np.bincount(y[mask]).argmax()) if mask.sum() > 0 else 0
    def predict_m2(Xin):
        return np.array([c2c[lbl] for lbl in m2.predict(Xin)])
else:
    m2 = MLPClassifier(
        hidden_layer_sizes=hidden_layers, activation=activation,
        max_iter=max_iter_mlp, learning_rate_init=learning_rate_init,
        random_state=random_state, early_stopping=True,
    )
    m2.fit(X2_train, y2_train)
    def predict_m2(Xin):
        return m2.predict(Xin)

h = 0.05
xx2, yy2 = np.meshgrid(
    np.arange(X2[:, 0].min()-0.8, X2[:, 0].max()+0.8, h),
    np.arange(X2[:, 1].min()-0.8, X2[:, 1].max()+0.8, h),
)
Z2 = predict_m2(np.c_[xx2.ravel(), yy2.ravel()]).reshape(xx2.shape)

fig_db, ax_db = plt.subplots(figsize=(7, 4.5))
ax_db.contourf(xx2, yy2, Z2, alpha=0.30, cmap=ListedColormap(PAL_BG))
for cls in range(len(class_names)):
    mask = y == cls
    ax_db.scatter(X2[mask, 0], X2[mask, 1], c=PAL_FG[cls], marker=MKRS[cls],
                  edgecolors="white", linewidths=0.5, s=60, label=class_names[cls])
ax_db.set_xlabel(f"{feat_x} (normalizado)", fontsize=10)
ax_db.set_ylabel(f"{feat_y} (normalizado)", fontsize=10)
ax_db.set_title(f"{model_choice} — Frontera de Decisión", fontsize=11)
ax_db.legend(loc="upper right", fontsize=8)
fig_db.tight_layout()
st.pyplot(fig_db)

# ── Gráfico de codo (solo K-Means) ───────────────────────────────────────────
if model_choice == "K-Means":
    st.markdown("---")
    st.subheader("Método del Codo — Selección de k")
    inertias = []
    k_range  = range(1, 11)
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    fig_elbow, ax_elbow = plt.subplots(figsize=(6, 3.5))
    ax_elbow.plot(k_range, inertias, "o-", color="#1a5cb5", linewidth=2, markersize=6)
    ax_elbow.axvline(n_clusters, color="#e63946", linestyle="--", linewidth=1.5,
                     label=f"k actual = {n_clusters}")
    ax_elbow.set_xlabel("Número de clústeres (k)", fontsize=10)
    ax_elbow.set_ylabel("Inercia", fontsize=10)
    ax_elbow.set_title("Método del Codo para K-Means", fontsize=11)
    ax_elbow.legend(fontsize=9)
    fig_elbow.tight_layout()
    st.pyplot(fig_elbow)

# ── Vista previa de datos ─────────────────────────────────────────────────────
with st.expander("📄 Vista previa de los datos"):
    st.dataframe(df)