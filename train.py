"""
train.py — Entrenamiento y validación del clasificador barco / no barco (ShipsNet).

Uso (PowerShell, desde la carpeta del proyecto):
    python train.py --data "C:/Users/<tu_usuario>/Downloads/archive/shipsnet/shipsnet"

Etapas (cada una genera evidencia para E2 / C1):
  1. Carga + extracción de descriptores (con y sin aumentación)
  2. Hold-out estratificado 80/20 que NO se toca hasta el final
  3. Ablación de descriptores (HOG, HOG+color, HOG+color+LBP)
  4. Comparación de modelos vistos en clase (SVM, RF, KNN, Bayes, Reg. logística)
  5. GridSearchCV (5-fold estratificado) de la arquitectura Scaler -> PCA -> SVM-RBF
  6. Validación cruzada con aumentación dentro de cada fold (sin fuga de datos)
  7. Evaluación en hold-out y entrenamiento del modelo final con todos los datos
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, ConfusionMatrixDisplay)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from features import ALL_BLOCKS, augment, extract_blocks, label_from_name, list_images, load_image

SEED = 42
N_AUG = 4  # original + fliplr + flipud + rot90


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def metrics(y, p):
    return {"accuracy": accuracy_score(y, p), "precision": precision_score(y, p, zero_division=0),
            "recall": recall_score(y, p, zero_division=0), "f1": f1_score(y, p, zero_division=0)}


def stack(blocks_list, names):
    return np.stack([np.concatenate([b[k] for k in names]) for b in blocks_list])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Carpeta con los PNG de ShipsNet (0__*.png / 1__*.png)")
    ap.add_argument("--out", default="model")
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--quick", action="store_true", help="Grilla reducida (prueba rápida)")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(exist_ok=True)
    report = {"fecha": datetime.now().isoformat(timespec="seconds")}

    # ------------------------------------------------------------------ 1. carga
    paths = [p for p in list_images(args.data) if label_from_name(p.name) is not None]
    if not paths:
        raise SystemExit("No encontré imágenes 0__*.png / 1__*.png en --data")
    y = np.array([label_from_name(p.name) for p in paths])
    log(f"{len(paths)} imágenes | barcos={int(y.sum())} | no barcos={int((y == 0).sum())}")

    t0 = time.time()
    blocks, blocks_aug = [], []  # blocks_aug[i] = lista de N_AUG dicts
    for i, p in enumerate(paths):
        views = augment(load_image(p))
        vb = [extract_blocks(v) for v in views]
        blocks.append(vb[0]); blocks_aug.append(vb)
        if (i + 1) % 500 == 0:
            log(f"  descriptores {i + 1}/{len(paths)}")
    report["tiempo_extraccion_s"] = round(time.time() - t0, 1)
    dims = {k: int(len(blocks[0][k])) for k in ALL_BLOCKS}
    report["dimensiones_descriptores"] = dims
    log(f"Dimensiones: {dims}")

    # --------------------------------------------------------------- 2. hold-out
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
    report["holdout"] = {"n_train": int(len(tr)), "n_test": int(len(te))}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    base_svm = Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="rbf", random_state=SEED))])

    # --------------------------------------------------------------- 3. ablación
    log("Ablación de descriptores (SVM-RBF por defecto, 5-fold)")
    ablation = {}
    for combo in [("hog",), ("color", "lbp"), ("hog", "color"), ("hog", "color", "lbp")]:
        Xc = stack([blocks[i] for i in tr], combo)
        s = cross_val_score(base_svm, Xc, y[tr], cv=cv, n_jobs=args.jobs)
        ablation["+".join(combo)] = {"media": float(s.mean()), "std": float(s.std())}
        log(f"  {'+'.join(combo):16s} acc={s.mean():.4f} ± {s.std():.4f}")
    report["ablacion"] = ablation
    best_combo = max(ablation, key=lambda k: ablation[k]["media"]).split("+")
    report["descriptores_elegidos"] = best_combo

    X = stack(blocks, best_combo)
    Xtr, ytr = X[tr], y[tr]

    # ------------------------------------------------------ 4. comparación modelos
    log("Comparación de modelos (5-fold)")
    models = {
        "SVM-RBF": base_svm,
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=args.jobs),
        "KNN (k=5)": Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(5))]),
        "Naive Bayes": Pipeline([("scaler", StandardScaler()), ("nb", GaussianNB())]),
        "Reg. logística": Pipeline([("scaler", StandardScaler()),
                                    ("lr", LogisticRegression(max_iter=3000, C=0.1))]),
    }
    comp = {}
    for name, m in models.items():
        t = time.time()
        s = cross_val_score(m, Xtr, ytr, cv=cv, n_jobs=args.jobs)
        comp[name] = {"media": float(s.mean()), "std": float(s.std()), "tiempo_s": round(time.time() - t, 1)}
        log(f"  {name:15s} acc={s.mean():.4f} ± {s.std():.4f}")
    report["comparacion_modelos"] = comp

    # ------------------------------------------------------------ 5. GridSearchCV
    log("GridSearchCV Scaler -> PCA -> SVM-RBF")
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("pca", PCA(random_state=SEED)),
                     ("svm", SVC(kernel="rbf", random_state=SEED))])
    grid = {"pca": ["passthrough", PCA(0.95, random_state=SEED), PCA(0.99, random_state=SEED)],
            "svm__C": [1, 10, 100],
            "svm__gamma": ["scale", 1e-4, 1e-3]}
    if args.quick:
        grid = {"pca": [PCA(0.95, random_state=SEED)], "svm__C": [10], "svm__gamma": ["scale"]}
    gs = GridSearchCV(pipe, grid, cv=cv, scoring="accuracy", n_jobs=args.jobs, refit=False)
    gs.fit(Xtr, ytr)
    bi = gs.best_index_
    best_params = gs.best_params_
    fold_scores = [float(gs.cv_results_[f"split{k}_test_score"][bi]) for k in range(5)]
    report["gridsearch"] = {
        "mejores_parametros": {k: str(v) for k, v in best_params.items()},
        "cv_media": float(gs.best_score_), "cv_std": float(gs.cv_results_["std_test_score"][bi]),
        "folds": fold_scores,
        "tabla": [{"params": {k: str(v) for k, v in p.items()}, "media": float(m), "std": float(s)}
                  for p, m, s in zip(gs.cv_results_["params"], gs.cv_results_["mean_test_score"],
                                     gs.cv_results_["std_test_score"])],
    }
    log(f"  mejor: {best_params} -> {gs.best_score_:.4f}")
    best_pipe = clone(pipe).set_params(**best_params)

    # ------------------------------------- 6. CV con aumentación dentro del fold
    log("CV con aumentación solo en los folds de entrenamiento")
    aug_scores = []
    for k, (a, b) in enumerate(cv.split(Xtr, ytr)):
        tr_i, va_i = tr[a], tr[b]
        Xa = stack([v for i in tr_i for v in blocks_aug[i]], best_combo)
        ya = np.repeat(y[tr_i], N_AUG)
        m = clone(best_pipe).fit(Xa, ya)
        aug_scores.append(float(accuracy_score(y[va_i], m.predict(X[va_i]))))
        log(f"  fold {k + 1}: {aug_scores[-1]:.4f}")
    use_aug = float(np.mean(aug_scores)) >= float(gs.best_score_)
    report["cv_aumentacion"] = {"folds": aug_scores, "media": float(np.mean(aug_scores)),
                                "std": float(np.std(aug_scores)), "usada_en_modelo_final": use_aug}
    cv_final = report["cv_aumentacion"] if use_aug else {"media": report["gridsearch"]["cv_media"],
                                                          "std": report["gridsearch"]["cv_std"],
                                                          "folds": fold_scores}
    report["cv_final"] = cv_final
    log(f"  aumentación {'SÍ' if use_aug else 'NO'} mejora -> se {'usa' if use_aug else 'descarta'}")

    def fit_on(ids, prob=False):
        if use_aug:
            Xf = stack([v for i in ids for v in blocks_aug[i]], best_combo); yf = np.repeat(y[ids], N_AUG)
        else:
            Xf, yf = X[ids], y[ids]
        m = clone(best_pipe)
        m.set_params(svm__probability=prob)
        return m.fit(Xf, yf)

    # ------------------------------------------------------------ 7. hold-out
    log("Evaluación en hold-out (20% nunca visto)")
    m_ho = fit_on(tr)
    p_ho = m_ho.predict(X[te])
    report["holdout"].update(metrics(y[te], p_ho))
    report["holdout"]["matriz_confusion"] = confusion_matrix(y[te], p_ho).tolist()
    log(f"  hold-out: {report['holdout']['accuracy']:.4f}")

    log("Entrenando modelo final con el 100% de los datos")
    final = fit_on(idx, prob=True)
    t = time.time()
    for _ in range(50):
        final.predict_proba(X[:1])
    report["latencia_modelo_ms"] = round((time.time() - t) / 50 * 1000, 2)

    joblib.dump({"pipeline": final, "blocks": best_combo, "threshold": 0.5}, out / "model.joblib",
                compress=3)
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # ------------------------------------------------------------- figuras
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay(np.array(report["holdout"]["matriz_confusion"]),
                           display_labels=["No barco", "Barco"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Hold-out"); fig.tight_layout(); fig.savefig(out / "cm_holdout.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    names = list(comp); vals = [comp[n]["media"] * 100 for n in names]
    ax.barh(names, vals, xerr=[comp[n]["std"] * 100 for n in names], color="#2f6f8f")
    ax.axvline(98, color="crimson", ls="--", label="Meta 98%"); ax.set_xlim(min(vals) - 2, 100)
    ax.set_xlabel("Accuracy CV 5-fold (%)"); ax.legend(); fig.tight_layout()
    fig.savefig(out / "comparacion_modelos.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3))
    names = list(ablation); vals = [ablation[n]["media"] * 100 for n in names]
    ax.barh(names, vals, color="#4a8f6a"); ax.set_xlim(min(vals) - 2, 100)
    ax.set_xlabel("Accuracy CV 5-fold (%)"); ax.set_title("Ablación de descriptores"); fig.tight_layout()
    fig.savefig(out / "ablacion.png", dpi=150); plt.close(fig)

    log(f"Listo. Modelo y reporte en ./{out}/")


if __name__ == "__main__":
    main()
