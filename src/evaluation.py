"""Ejecución y registro de experimentos de clasificación.

Regla de oro que atraviesa todo el módulo: **los hiperparámetros se eligen
usando solo las n muestras de entrenamiento**, por validación cruzada interna.
Nunca se mira el test para elegir nada. Si la SVM afinara su gamma contra el
test, la curva de precisión frente al tamaño de entrenamiento —el experimento
central del trabajo— quedaría inflada de forma distinta a cada n, y la
comparación contra la CNN dejaría de significar nada.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from src.classifiers import AffinePCAClassifier
from src.repro import environment_report

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Rejillas de búsqueda. La de la SVM cubre varios órdenes de magnitud porque la
# escala adecuada de gamma depende de la dimensión del descriptor, que va de
# 784 (píxeles) a 3472 (scattering de orden 2).
SVM_GRID = {
    "svc__C": [1.0, 10.0, 100.0, 1000.0],
    "svc__gamma": ["scale", 1e-4, 1e-3, 1e-2],
}

# Dos decisiones deliberadas sobre la búsqueda en rejilla:
#
# n_jobs=1: con el descriptor de orden 2 el array de features ocupa cientos de
# MB, y los procesos paralelos de joblib agotaban la memoria en esta máquina.
# Cada ajuste tarda segundos, así que paralelizar no compensa el riesgo.
#
# error_score="raise": por defecto, un ajuste que falla se convierte en un score
# `nan` y GridSearchCV sigue adelante eligiendo entre los supervivientes, con
# un simple warning. Eso es justo lo que ocurrió al desarrollar esto: cuatro de
# seis valores de d fallaban en silencio. Preferimos que reviente.
GRID_SEARCH_KWARGS = {"n_jobs": 1, "error_score": "raise"}


@dataclass
class Result:
    """Un experimento: qué se corrió, con qué datos, y qué salió."""

    model: str
    descriptor: str
    train_size: int
    seed: int
    error_rate: float
    best_params: dict[str, Any] = field(default_factory=dict)
    n_features: int = 0
    fit_seconds: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "descriptor": self.descriptor,
            "train_size": self.train_size,
            "seed": self.seed,
            "error_rate": self.error_rate,
            "best_params": self.best_params,
            "n_features": self.n_features,
            "fit_seconds": self.fit_seconds,
            "extra": self.extra,
        }
        return payload


def evaluate_svm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    descriptor: str,
    seed: int,
    cv: int = 3,
    grid: dict | None = None,
) -> Result:
    """SVM con kernel RBF, hiperparámetros por validación cruzada interna.

    Los coeficientes se llevan a [-1, 1] con el escalador ajustado solo sobre
    entrenamiento, que es la normalización que el paper indica para la SVM.
    """
    grid = grid or SVM_GRID
    pipeline = Pipeline(
        [
            ("scaler", MinMaxScaler(feature_range=(-1, 1))),
            ("svc", SVC(kernel="rbf", random_state=seed)),
        ]
    )

    # cv no puede exceder el mínimo de muestras por clase; con n=300 hay 30 por
    # clase, así que 3 pliegues siempre caben, pero conviene no asumirlo.
    _, counts = np.unique(y_train, return_counts=True)
    folds = int(min(cv, counts.min()))

    search = GridSearchCV(pipeline, grid, cv=folds, refit=True, **GRID_SEARCH_KWARGS)

    start = time.perf_counter()
    search.fit(x_train, y_train)
    elapsed = time.perf_counter() - start

    error = 1.0 - search.score(x_test, y_test)
    return Result(
        model="svm",
        descriptor=descriptor,
        train_size=len(x_train),
        seed=seed,
        error_rate=error,
        best_params={k: str(v) for k, v in search.best_params_.items()},
        n_features=x_train.shape[1],
        fit_seconds=elapsed,
    )


def evaluate_affine_pca(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    descriptor: str,
    seed: int,
    dimensions: list[int] | None = None,
    cv: int = 3,
) -> Result:
    """Clasificador generativo de espacios afines, con d por validación cruzada.

    El rango de d se recorta al número de muestras por clase: no tiene sentido
    pedir 140 direcciones principales a una clase con 30 ejemplos.
    """
    _, counts = np.unique(y_train, return_counts=True)
    per_class = int(counts.min())
    folds = int(min(cv, per_class))

    # Con validación cruzada de `folds` pliegues, cada ajuste ve una fracción
    # (folds-1)/folds de la clase, y d no puede superar ese rango.
    usable = max(1, (per_class * (folds - 1)) // folds - 1)
    candidates = dimensions or [3, 5, 10, 20, 40, 80, 140, 200]
    candidates = sorted({int(min(d, usable)) for d in candidates})

    search = GridSearchCV(
        AffinePCAClassifier(),
        {"n_components": candidates},
        cv=folds,
        refit=True,
        **GRID_SEARCH_KWARGS,
    )

    start = time.perf_counter()
    search.fit(x_train, y_train)
    elapsed = time.perf_counter() - start

    error = 1.0 - search.score(x_test, y_test)
    sigma_d2, lambda_d = search.best_estimator_.approximation_errors(x_test, y_test)

    return Result(
        model="affine_pca",
        descriptor=descriptor,
        train_size=len(x_train),
        seed=seed,
        error_rate=error,
        best_params={k: str(v) for k, v in search.best_params_.items()},
        n_features=x_train.shape[1],
        fit_seconds=elapsed,
        extra={"sigma_d2": sigma_d2, "lambda_d": lambda_d},
    )


def save_results(results: list[Result], name: str) -> Path:
    """Guarda resultados junto a la instantánea del entorno que los produjo."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"

    payload = {
        "environment": environment_report().to_dict(),
        "results": [r.to_dict() for r in results],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_results(name: str) -> list[dict[str, Any]]:
    """Lee resultados previamente guardados."""
    path = RESULTS_DIR / f"{name}.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))["results"]
