# Detección de barcos en imágenes aéreas (ShipsNet) — Proyecto 2 IA, UMNG

Clasificador binario barco / no barco (80×80 RGB) para el sistema de percepción de un UAV
de inspección portuaria (caso Rotterdam). Descriptores HOG + color + LBP, arquitectura
StandardScaler → PCA → SVM-RBF optimizada con GridSearchCV y validación cruzada estratificada 5-fold.

## Estructura

```
proyecto/
├── features.py      # preprocesamiento + descriptores (idéntico en entrenamiento e inferencia)
├── train.py         # ablación, comparación de modelos, GridSearch, CV, hold-out, modelo final
├── app.py           # UI Streamlit: carga desde carpeta, etiquetado, inferencia y métricas en vivo
├── requirements.txt
└── model/           # lo genera train.py (model.joblib, report.json, figuras)
```

## 1. Instalar (PowerShell)

```powershell
cd proyecto
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. Entrenar

```powershell
python train.py --data "$env:USERPROFILE\Downloads\archive\shipsnet\shipsnet"
```

Ajusta la ruta a la carpeta que contiene los PNG `0__*.png` y `1__*.png`.
Tarda unos minutos. Al terminar, `model/report.json` contiene todas las cifras para el informe
y `model/*.png` las figuras (comparación de modelos, ablación, matriz de confusión hold-out).

## 3. Prueba en vivo

```powershell
streamlit run app.py
```

1. Copia las imágenes de test a una carpeta (por defecto `./test`) y pulsa **Cargar carpeta**.
2. Pulsa **Ejecutar inferencia en vivo**: las métricas se actualizan mientras avanza.
3. Etiqueta cada imagen (Barco / No barco); accuracy, precisión, recall, F1 y la matriz de confusión se recalculan al instante.
4. Pestaña **Validación cruzada vs. en vivo**: contraste CV ↔ hold-out ↔ test ciego.
5. **Descargar resultados (CSV)** como evidencia.

Para ensayar con ShipsNet, activa *Tomar etiqueta del nombre (0__/1__)*. En la prueba ciega déjalo apagado.
