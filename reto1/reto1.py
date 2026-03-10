import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score)
from matplotlib.colors import ListedColormap

st.set_page_config(page_title="Clasificador de Iris", page_icon="🌸", layout="wide")
st.title("Clasificador de Iris")
st.markdown("Comparacion de **Modelo Lineal**, **Árbol de Decisión** y **Bosque Aleatorio** en el dataset de Iris.")

class LinearRegressionClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        self.n_classes_ = len(np.unique(y))
        self._reg = LinearRegression(fit_intercept=self.fit_intercept)
        self._reg.fit(X, y)
        self.coef_ = self._reg.coef_
        self.intercept_ = self._reg.intercept_
        return self

    def predict(self, X):
        raw = self._reg.predict(X)
        return np.clip(np.round(raw).astype(int), 0, self.n_classes_ - 1)

    def predict_raw(self, X):
        return self._reg.predict(X)

st.sidebar.header("⚙️ Configuración")

model_choice = st.sidebar.selectbox(
    "Cambiar el modelo",
    ["Lineal", "Arbol de decision", "Bosque aleatorio"],
)
test_size    = st.sidebar.slider("Tamaño del conjunto de prueba", 0.1, 0.5, 0.2, 0.05)
random_state = st.sidebar.number_input("Seed", 0, 999, 42, 1)

st.sidebar.markdown("---")
st.sidebar.subheader("Hiperparametros")

if model_choice == "Lineal":
    fit_intercept = st.sidebar.checkbox("Ajustar intercepto", value=True)
elif model_choice == "Arbol de decision":
    max_depth         = st.sidebar.slider("Profundidad maxima", 1, 10, 3)
    min_samples_split = st.sidebar.slider("Minimo de muestras para dividir", 2, 20, 2)
else:
    n_estimators  = st.sidebar.slider("Numero de arboles", 10, 300, 100, 10)
    max_depth_rf  = st.sidebar.slider("Profundidad maxima", 1, 15, 5)

# ── Load data — auto-detect CSV next to script ────────────────────────────────
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

# ── Prepare features ──────────────────────────────────────────────────────────
feature_cols = [c for c in df.columns if c != "Species"]
X  = df[feature_cols].values
le = LabelEncoder()
y  = le.fit_transform(df["Species"].values)
class_names = le.classes_

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y
)

# ── Build model ───────────────────────────────────────────────────────────────
if model_choice == "Lineal":
    model = LinearRegressionClassifier(fit_intercept=fit_intercept)
elif model_choice == "Arbol de decision":
    model = DecisionTreeClassifier(
        max_depth=max_depth, min_samples_split=min_samples_split, random_state=random_state
    )
else:
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth_rf, random_state=random_state
    )

model.fit(X_train, y_train)
y_pred    = model.predict(X_test)
accuracy  = accuracy_score(y_test, y_pred)
cv_scores = cross_val_score(model, X, y, cv=5)

# ── Top metrics row ───────────────────────────────────────────────────────────
st.markdown("---")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Modelo",            model_choice)
c2.metric("Exactitud de la prueba",    f"{accuracy:.2%}")
c3.metric("Media de validacion cruzada", f"{cv_scores.mean():.2%}")
c4.metric("Desviacion de validacion cruzada",           f"± {cv_scores.std():.2%}")
st.markdown("---")

# ── Confusion matrix  +  model-specific viz ───────────────────────────────────
left, right = st.columns(2)

with left:
    st.subheader("Matriz de confusion")
    fig_cm, ax_cm = plt.subplots(figsize=(4, 3.5))
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, y_pred), display_labels=class_names
    ).plot(ax=ax_cm, colorbar=False, cmap="Blues")
    ax_cm.set_xticklabels(class_names, rotation=20, ha="right", fontsize=8)
    ax_cm.set_yticklabels(class_names, fontsize=8)
    fig_cm.tight_layout()
    st.pyplot(fig_cm)

with right:
    if model_choice == "Arbol de decision":
        st.subheader("Arbol de decision")
        fig_tree, ax_tree = plt.subplots(figsize=(6, 4))
        plot_tree(model, feature_names=feature_cols, class_names=class_names,
                  filled=True, rounded=True, fontsize=7, ax=ax_tree)
        fig_tree.tight_layout()
        st.pyplot(fig_tree)

    elif model_choice == "Bosque aleatorio":
        st.subheader("Importancia de variables")
        importances = model.feature_importances_
        indices     = np.argsort(importances)[::-1]
        colors      = ["#5c85d6", "#e07b54", "#6dbf67", "#c46abf"]
        fig_fi, ax_fi = plt.subplots(figsize=(5, 3.5))
        ax_fi.bar(range(len(feature_cols)), importances[indices],
                  color=[colors[i % len(colors)] for i in range(len(feature_cols))])
        ax_fi.set_xticks(range(len(feature_cols)))
        ax_fi.set_xticklabels([feature_cols[i] for i in indices],
                               rotation=20, ha="right", fontsize=9)
        ax_fi.set_ylabel("Importancia")
        ax_fi.set_title("Bosque aleatorio – Importancia de variables")
        fig_fi.tight_layout()
        st.pyplot(fig_fi)

    else:  # Lineal
        st.subheader("Modelo lineal — Observado vs Predicho (1 característica)")
        feat_sel = st.selectbox("Caracteristica a graficar", feature_cols, index=2, key="lr_feat")
        f_idx    = feature_cols.index(feat_sel)
        Xf       = X[:, f_idx]

        lr1d      = LinearRegression(fit_intercept=fit_intercept).fit(Xf.reshape(-1, 1), y)
        slope     = lr1d.coef_[0]
        intercept = lr1d.intercept_
        x_line    = np.linspace(Xf.min(), Xf.max(), 200)
        y_line    = lr1d.predict(x_line.reshape(-1, 1))

        pal_fg  = ["#1a5cb5", "#b54a1a", "#1ab53c"]
        markers = ["o", "s", "^"]

        fig_lr, ax_lr = plt.subplots(figsize=(6, 4.5))
        for cls in range(len(class_names)):
            mask = y == cls
            ax_lr.scatter(Xf[mask], y[mask], c=pal_fg[cls], marker=markers[cls],
                          edgecolors="white", linewidths=0.5, s=55,
                          label=class_names[cls], zorder=3)

        ax_lr.plot(x_line, y_line, color="#222", linewidth=2,
                   label=f"Y = {intercept:.2f} + {slope:.2f}·X")

        ex_idx = len(Xf) // 2
        xi_val = Xf[ex_idx]
        yi_obs = float(y[ex_idx])
        yp_val = float(lr1d.predict([[xi_val]])[0])

        ax_lr.plot([xi_val, xi_val], [yp_val, yi_obs],
                   color="#e63946", linewidth=1.5, linestyle="--", zorder=4)
        ax_lr.scatter([xi_val], [yi_obs], color="#e63946", s=90, zorder=5,
                      label="yᵢ observado")
        ax_lr.scatter([xi_val], [yp_val], color="#222", s=90, marker="D",
                      zorder=5, label="Ypᵢ predicho")

        ax_lr.annotate("yᵢ (obs)",   xy=(xi_val, yi_obs),
                       xytext=(xi_val+0.15, yi_obs+0.05), fontsize=8, color="#e63946")
        ax_lr.annotate("Ypᵢ (pred)", xy=(xi_val, yp_val),
                       xytext=(xi_val+0.15, yp_val-0.12), fontsize=8)
        ax_lr.annotate("Error\naleatorio εᵢ",
                       xy=(xi_val, (yi_obs+yp_val)/2),
                       xytext=(xi_val-1.1, (yi_obs+yp_val)/2),
                       fontsize=7.5, color="#e63946",
                       arrowprops=dict(arrowstyle="-[,widthB=1.0", color="#e63946", lw=1))
        mid = len(x_line) // 2
        ax_lr.annotate(f"Y = θ₁ + θ₂X\nPendiente = tan θ = {slope:.2f}",
                       xy=(x_line[mid], y_line[mid]),
                       xytext=(x_line[mid]+0.3, y_line[mid]-0.4), fontsize=8,
                       arrowprops=dict(arrowstyle="->", color="#555", lw=0.8))
        ax_lr.annotate(f"Intercepto θ₁ = {intercept:.2f}",
                       xy=(Xf.min(), intercept),
                       xytext=(Xf.min()+0.3, intercept-0.35), fontsize=8, color="#555",
                       arrowprops=dict(arrowstyle="->", color="#555", lw=0.8))

        ax_lr.set_xlabel(feat_sel, fontsize=10)
        ax_lr.set_ylabel("Etiqueta de clase (0 / 1 / 2)", fontsize=10)
        ax_lr.set_title("Lineal — Observado vs Predicho", fontsize=11)
        ax_lr.set_yticks([0, 1, 2])
        ax_lr.set_yticklabels(class_names, fontsize=8)
        ax_lr.legend(fontsize=7.5, loc="upper left")
        fig_lr.tight_layout()
        st.pyplot(fig_lr)

        st.markdown("**Todas las coeficientes del modelo**")
        coef_df = pd.DataFrame({"Coeficiente (pendiente)": dict(zip(feature_cols, model.coef_))})
        st.dataframe(coef_df.style.background_gradient(cmap="RdYlGn"))

# ── Classification report ─────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Reporte de clasificacion")
report    = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
report_df = pd.DataFrame(report).T.round(3)
st.dataframe(report_df.style.background_gradient(
    cmap="Blues", subset=["precision", "recall", "f1-score"]))

# ── 2-D decision boundary ─────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Limite de decision (2 caracteristicas)")
feat_x = st.selectbox("Eje X", feature_cols, index=0)
feat_y = st.selectbox("Eje Y", feature_cols, index=2)
idx_x, idx_y = feature_cols.index(feat_x), feature_cols.index(feat_y)

X2 = X[:, [idx_x, idx_y]]
X2_train, _, y2_train, _ = train_test_split(
    X2, y, test_size=test_size, random_state=random_state, stratify=y
)

if model_choice == "Lineal":
    m2 = LinearRegressionClassifier(fit_intercept=fit_intercept)
elif model_choice == "Arbol de decision":
    m2 = DecisionTreeClassifier(
        max_depth=max_depth, min_samples_split=min_samples_split, random_state=random_state
    )
else:
    m2 = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth_rf, random_state=random_state
    )
m2.fit(X2_train, y2_train)

h = 0.02
xx, yy = np.meshgrid(
    np.arange(X2[:, 0].min() - 0.5, X2[:, 0].max() + 0.5, h),
    np.arange(X2[:, 1].min() - 0.5, X2[:, 1].max() + 0.5, h),
)
Z = m2.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

pal_bg = ["#aec6f0", "#f0c8ae", "#aef0b4"]
pal_fg = ["#1a5cb5", "#b54a1a", "#1ab53c"]
mkrs   = ["o", "s", "^"]

fig_db, ax_db = plt.subplots(figsize=(7, 4.5))
ax_db.contourf(xx, yy, Z, alpha=0.35, cmap=ListedColormap(pal_bg))
for cls in range(len(class_names)):
    mask = y == cls
    ax_db.scatter(X2[mask, 0], X2[mask, 1], c=pal_fg[cls], marker=mkrs[cls],
                  edgecolors="white", linewidths=0.5, s=60, label=class_names[cls])
ax_db.set_xlabel(feat_x)
ax_db.set_ylabel(feat_y)
ax_db.set_title(f"{model_choice} — Limite de decision")
ax_db.legend(loc="upper right", fontsize=8)
fig_db.tight_layout()
st.pyplot(fig_db)

# ── Raw data preview ──────────────────────────────────────────────────────────
with st.expander("📊 Ver datos crudos"):
    st.dataframe(df)