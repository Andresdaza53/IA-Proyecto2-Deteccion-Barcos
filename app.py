"""
app.py — UI de evaluación en vivo (E1, E3, E4).

Ejecutar:  streamlit run app.py
"""
import io
import json
import math
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix

from features import extract_features, label_from_name, list_images, load_image

MODEL_DIR = Path(__file__).parent / "model"
LABELS = {1: "Barco", 0: "No barco"}
OPTIONS = ["Sin etiqueta", "Barco", "No barco"]
OPT2Y = {"Sin etiqueta": None, "Barco": 1, "No barco": 0}
Y2OPT = {None: "Sin etiqueta", 1: "Barco", 0: "No barco"}
PAGE = 30

st.set_page_config(page_title="Detección de barcos — UAV Rotterdam", page_icon="🚢", layout="wide")
st.markdown("""
<style>
  :root { --sea:#0f3d56; --hull:#e2a33b; --ok:#2e7d5b; --bad:#b3362c; }
  h1, h2, h3 { color: var(--sea); }
  .tag { display:inline-block; padding:2px 8px; border-radius:4px; font-size:.85rem; color:white; }
  .ship { background: var(--hull); color:#1b1b1b; } .noship { background: var(--sea); }
  .hit { border-left:4px solid var(--ok); padding-left:6px; }
  .miss { border-left:4px solid var(--bad); padding-left:6px; }
  .idle { border-left:4px solid #9aa7b0; padding-left:6px; }
</style>""", unsafe_allow_html=True)


# ----------------------------------------------------------------- recursos
@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_DIR / "model.joblib")
    rep = json.loads((MODEL_DIR / "report.json").read_text(encoding="utf-8"))
    return bundle, rep


def ss(key, default):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


try:
    bundle, report = load_model()
except FileNotFoundError:
    st.error("No encontré model/model.joblib. Ejecuta primero: python train.py --data <carpeta shipsnet>")
    st.stop()

items = ss("imgs", [])      # [{name, img}]
preds = ss("preds", {})      # name -> {"prob", "pred", "ms"}
labels = ss("labels", {})    # name -> 0/1/None

# ------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Imágenes de test")
    source = st.radio("Origen", ["Carpeta local", "Subir archivos"], horizontal=True)
    auto_label = st.checkbox("Tomar etiqueta del nombre (0__/1__) si existe", value=False,
                             help="Útil para ensayar con ShipsNet. En la prueba ciega, déjalo apagado y etiqueta a mano.")
    loaded = None
    if source == "Carpeta local":
        folder = st.text_input("Ruta de la carpeta", value=str(Path.cwd() / "test"))
        if st.button("Cargar carpeta", width="stretch"):
            if Path(folder).is_dir():
                loaded = [(p.name, load_image(p)) for p in list_images(folder)]
            else:
                st.error("Esa ruta no existe o no es una carpeta.")
    else:
        ups = st.file_uploader("Imágenes", type=["png", "jpg", "jpeg", "bmp", "tif"], accept_multiple_files=True)
        if st.button("Cargar archivos", width="stretch") and ups:
            loaded = [(u.name, load_image(io.BytesIO(u.getvalue()))) for u in ups]

    if loaded is not None:
        st.session_state["imgs"] = [{"name": n, "img": im} for n, im in loaded]
        st.session_state.preds = {}
        st.session_state.labels = {n: (label_from_name(n) if auto_label else None) for n, _ in loaded}
        st.session_state.page = 0
        for n, _ in loaded:
            st.session_state.pop(f"lbl_{n}", None)
        st.rerun()

    st.divider()
    thr = st.slider("Umbral de decisión P(barco)", 0.05, 0.95, float(bundle.get("threshold", 0.5)), 0.05)
    st.caption(f"Descriptores: {' + '.join(bundle['blocks']).upper()}")
    st.caption(f"Modelo: {' → '.join(n for n, _ in bundle['pipeline'].steps)}")

items, preds, labels = st.session_state["imgs"], st.session_state.preds, st.session_state.labels
for _it in items:  # sincroniza etiquetas de los widgets ANTES de calcular métricas
    _k = f"lbl_{_it['name']}"
    if _k in st.session_state:
        labels[_it["name"]] = OPT2Y[st.session_state[_k]]


def predict(item):
    t = time.perf_counter()
    x = extract_features(item["img"], bundle["blocks"])[None, :]
    prob = float(bundle["pipeline"].predict_proba(x)[0, 1])
    return {"prob": prob, "ms": (time.perf_counter() - t) * 1000}


def current_rows():
    rows = []
    for it in items:
        n = it["name"]; p = preds.get(n)
        rows.append({"archivo": n, "etiqueta": labels.get(n),
                     "prob_barco": p["prob"] if p else None,
                     "prediccion": (int(p["prob"] >= thr) if p else None),
                     "latencia_ms": p["ms"] if p else None})
    return pd.DataFrame(rows)


def compute_metrics(df):
    d = df.dropna(subset=["etiqueta", "prediccion"])
    if d.empty:
        return None
    y, p = d["etiqueta"].astype(int).values, d["prediccion"].astype(int).values
    cm = confusion_matrix(y, p, labels=[0, 1]); tn, fp, fn, tp = cm.ravel()
    acc = (tp + tn) / len(y)
    return {"n": len(y), "acc": acc, "prec": tp / (tp + fp) if tp + fp else 0.0,
            "rec": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
            "spec": tn / (tn + fp) if tn + fp else 0.0, "cm": cm}


def penalty(acc):
    return 0.0 if acc >= 0.98 else 0.5 * math.ceil((0.98 - acc) * 100 / 2 - 1e-9)


def plot_cm(cm):
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, int(v), ha="center", va="center", fontsize=14,
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["No barco", "Barco"]); ax.set_yticks([0, 1], ["No barco", "Barco"])
    ax.set_xlabel("Predicción"); ax.set_ylabel("Real"); fig.tight_layout()
    return fig


def render_metrics(box):
    df = current_rows(); m = compute_metrics(df)
    with box.container():
        c = st.columns(6)
        c[0].metric("Imágenes", len(items))
        c[1].metric("Etiquetadas / inferidas", f"{int(df['etiqueta'].notna().sum())} / {int(df['prediccion'].notna().sum())}")
        if m:
            c[2].metric("Accuracy en vivo", f"{m['acc'] * 100:.2f} %",
                        f"{(m['acc'] - report['cv_final']['media']) * 100:+.2f} pp vs CV")
            c[3].metric("Precisión", f"{m['prec'] * 100:.2f} %")
            c[4].metric("Recall", f"{m['rec'] * 100:.2f} %")
            c[5].metric("F1", f"{m['f1'] * 100:.2f} %")
            l, r = st.columns([1, 2])
            fig = plot_cm(m["cm"]); l.pyplot(fig); plt.close(fig)
            pen = penalty(m["acc"])
            r.markdown(f"**Evaluadas:** {m['n']} imágenes con etiqueta y predicción  \n"
                       f"**Especificidad:** {m['spec'] * 100:.2f} %  \n"
                       f"**Meta ≥ 98 %:** {'cumplida ✅' if m['acc'] >= 0.98 else f'no cumplida — penalización estimada {pen}'}  \n"
                       f"**Latencia media por imagen:** {df['latencia_ms'].mean():.1f} ms "
                       f"(≈ {1000 / max(df['latencia_ms'].mean(), 1e-6):.0f} FPS en este equipo)")
        else:
            c[2].info("Etiqueta imágenes y ejecuta la inferencia para ver métricas.")


# ---------------------------------------------------------------------- main
st.title("Clasificación barco / no barco para inspección portuaria con UAV")
tab_live, tab_cv = st.tabs(["Evaluación en vivo", "Validación cruzada vs. en vivo"])

with tab_live:
    if not items:
        st.info("Carga la carpeta de test desde el panel izquierdo para empezar.")
    else:
        b1, b2, b3 = st.columns([1, 1, 2])
        run = b1.button("▶ Ejecutar inferencia en vivo", type="primary", width="stretch")
        if b2.button("Borrar predicciones", width="stretch"):
            st.session_state.preds = {}; st.rerun()
        metrics_box = st.empty()

        if run:
            bar = st.progress(0.0, text="Inferencia…")
            for i, it in enumerate(items):
                preds[it["name"]] = predict(it)
                if i % 5 == 0 or i == len(items) - 1:   # refresco en tiempo real
                    bar.progress((i + 1) / len(items), text=f"Inferencia {i + 1}/{len(items)}")
                    render_metrics(metrics_box)
            bar.empty()
        render_metrics(metrics_box)

        st.subheader("Etiquetado")
        flt = st.radio("Mostrar", ["Todas", "Sin etiqueta", "Errores"], horizontal=True)
        view = items
        if flt == "Sin etiqueta":
            view = [it for it in items if labels.get(it["name"]) is None]
        elif flt == "Errores":
            view = [it for it in items if it["name"] in preds and labels.get(it["name"]) is not None
                    and int(preds[it["name"]]["prob"] >= thr) != labels[it["name"]]]
        pages = max(1, math.ceil(len(view) / PAGE))
        page = st.number_input(f"Página (de {pages})", 1, pages, min(ss("page", 0) + 1, pages)) - 1
        st.session_state.page = page

        cols = st.columns(6)
        for k, it in enumerate(view[page * PAGE:(page + 1) * PAGE]):
            n = it["name"]; col = cols[k % 6]
            p = preds.get(n); lab = labels.get(n)
            cls = "idle"
            if p is not None and lab is not None:
                cls = "hit" if int(p["prob"] >= thr) == lab else "miss"
            with col:
                st.image((it["img"] * 255).astype(np.uint8), width="stretch")
                if p:
                    y_hat = int(p["prob"] >= thr)
                    st.markdown(f"<div class='{cls}'><span class='tag {'ship' if y_hat else 'noship'}'>"
                                f"{LABELS[y_hat]}</span> {p['prob'] * 100:.1f}%</div>", unsafe_allow_html=True)
                choice = st.radio(n[:22], OPTIONS, index=OPTIONS.index(Y2OPT[lab]), key=f"lbl_{n}",
                                  label_visibility="visible")
                labels[n] = OPT2Y[choice]

        df = current_rows()
        st.download_button("Descargar resultados (CSV)", df.to_csv(index=False).encode("utf-8"),
                           "resultados_evaluacion_en_vivo.csv", "text/csv")

with tab_cv:
    cvf = report["cv_final"]; ho = report["holdout"]
    m = compute_metrics(current_rows()) if items else None
    c = st.columns(3)
    c[0].metric("CV 5-fold (entrenamiento)", f"{cvf['media'] * 100:.2f} % ± {cvf['std'] * 100:.2f}")
    c[1].metric("Hold-out 20 %", f"{ho['accuracy'] * 100:.2f} %")
    c[2].metric("En vivo (test ciego)", f"{m['acc'] * 100:.2f} %" if m else "—")

    fig, ax = plt.subplots(figsize=(6, 3))
    names = [f"Fold {i + 1}" for i in range(len(cvf["folds"]))] + ["Hold-out"] + (["En vivo"] if m else [])
    vals = [v * 100 for v in cvf["folds"]] + [ho["accuracy"] * 100] + ([m["acc"] * 100] if m else [])
    colors = ["#2f6f8f"] * len(cvf["folds"]) + ["#4a8f6a"] + (["#e2a33b"] if m else [])
    ax.bar(names, vals, color=colors); ax.axhline(98, color="crimson", ls="--", label="Meta 98 %")
    ax.set_ylim(min(vals + [98]) - 2, 100.5); ax.set_ylabel("Accuracy (%)"); ax.legend(loc="lower right")
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    if m:
        lo = cvf["media"] - 2 * cvf["std"]
        hi = min(cvf["media"] + 2 * cvf["std"], 1.0)
        rango = f"{lo * 100:.2f} % – {hi * 100:.2f} %"
        if m["acc"] > hi:
            st.markdown(f"Intervalo esperado por CV (media ± 2σ): **{rango}**. El resultado en vivo está "
                        f"**por encima** del rango: rindió mejor de lo esperado. Con pocas imágenes, o con "
                        f"imágenes vistas en el entrenamiento, este valor puede ser optimista.")
        elif m["acc"] >= lo:
            st.markdown(f"Intervalo esperado por CV (media ± 2σ): **{rango}**. El resultado en vivo "
                        f"**cae dentro** del rango: el modelo generaliza como predijo la validación cruzada.")
        else:
            st.markdown(f"Intervalo esperado por CV (media ± 2σ): **{rango}**. El resultado en vivo está "
                        f"**por debajo** del rango: revisa el filtro *Errores* para ver si hay cambio de "
                        f"dominio (imágenes distintas a las de entrenamiento).")

    with st.expander("Detalle de la optimización (E2)"):
        st.write("**Mejores hiperparámetros:**", report["gridsearch"]["mejores_parametros"])
        st.write("**Descriptores elegidos:**", report["descriptores_elegidos"],
                 "| dimensiones:", report["dimensiones_descriptores"])
        st.write("**Aumentación usada:**", report["cv_aumentacion"]["usada_en_modelo_final"])
        st.dataframe(pd.DataFrame(report["ablacion"]).T.rename_axis("descriptores"))
        st.dataframe(pd.DataFrame(report["comparacion_modelos"]).T.rename_axis("modelo"))
        for f in ["comparacion_modelos.png", "ablacion.png", "cm_holdout.png"]:
            if (MODEL_DIR / f).exists():
                st.image(str(MODEL_DIR / f))
