"""Clasificador generativo de espacios afines (Bruna y Mallat, sección 4.1).

La idea: para cada clase k se ajusta un espacio afín A_{d,k} de dimensión d,
generado por el centroide de la clase y sus d direcciones principales. Una
muestra se asigna a la clase cuyo espacio afín la aproxima mejor, es decir la
que minimiza la norma del residuo de la proyección ortogonal

    ||S x - P_{A_{d,k}}(S x)|| .

Es un modelo generativo: cada clase se estima por separado y nunca se estiman
términos cruzados entre clases. Ahí está la razón de que gane a la SVM con pocas
muestras, y de que la pierda cuando hay muchas, con n pequeño, los términos de
varianza al estimar covarianzas cruzadas dominan al sesgo del modelo rígido.

El paper también lo lee como un discriminante cuadrático robusto en el que los
autovalores de la covarianza inversa se cuantizan groseramente a 0 dentro del
subespacio y a 1 en su ortogonal, lo cual se justifica porque con pocas muestras
esos autovalores se estiman mal.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class AffinePCAClassifier(BaseEstimator, ClassifierMixin):
    """Clasificador por distancia a un espacio afín por clase.

    Parameters
    ----------
    n_components:
        Dimensión d de los espacios afines. Es el único hiperparámetro y se
        ajusta por validación cruzada; en el paper crece con el tamaño del
        conjunto de entrenamiento (d=5 con 300 muestras, d=140 con 40000),
        porque con más datos se pueden estimar más direcciones con fiabilidad.
    """

    def __init__(self, n_components: int = 20):
        self.n_components = n_components

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AffinePCAClassifier":
        X = np.asarray(X)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        self.means_ = []
        self.bases_ = []

        for cls in self.classes_:
            # Se convierte a float64 clase a clase, no el conjunto entero: con
            # 60000 muestras de 3472 dimensiones una copia completa serían
            # 1.6 GB, mientras que una clase suelta ocupa la décima parte.
            members = np.asarray(X[y == cls], dtype=np.float64)
            mean = members.mean(axis=0)
            centered = members - mean

            # d no puede exceder el número de direcciones que los datos de la
            # clase determinan; con muy pocas muestras por clase esto recorta.
            max_rank = min(centered.shape) - 1
            d = int(np.clip(self.n_components, 1, max(max_rank, 1)))

            # Las direcciones principales son los autovectores de la covarianza,
            # que son los vectores singulares por la derecha de los datos
            # centrados. Se obtienen por SVD, sin formar la covarianza.
            _, _, vt = np.linalg.svd(centered, full_matrices=False)

            self.means_.append(mean)
            self.bases_.append(vt[:d])

        return self

    def n_parameters(self) -> int:
        """Números estimados a partir de los datos: centroides más bases.

        Existe para poder comparar honestamente con el conteo de parámetros de
        la CNN. El resultado suele sorprender —este clasificador maneja más
        números que la red— y esa incomodidad es informativa: la ventaja del
        scattering con pocos datos no viene de tener menos parámetros, sino de
        cómo se estiman. Aquí cada clase se resuelve por separado y en forma
        cerrada mediante una SVD; allí todos los pesos se ajustan a la vez por
        descenso de gradiente sobre un objetivo conjunto.
        """
        return sum(mean.size + basis.size for mean, basis in zip(self.means_, self.bases_))

    def _residuals(self, X: np.ndarray, chunk_size: int = 2048) -> np.ndarray:
        """Norma al cuadrado del residuo de proyección, por muestra y clase.

        Se procesa por bloques de filas: con el descriptor de orden 2 (3472
        dimensiones) y 10000 imágenes de test, una sola copia en float64 ocupa
        265 MB, y hacen falta varias a la vez.
        """
        X = np.asarray(X)
        residuals = np.empty((X.shape[0], len(self.classes_)))

        for start in range(0, X.shape[0], chunk_size):
            block = np.asarray(X[start : start + chunk_size], dtype=np.float64)

            for k, (mean, basis) in enumerate(zip(self.means_, self.bases_)):
                centered = block - mean
                # Teorema de Pitágoras: el residuo ortogonal es la norma total
                # menos la parte que cae dentro del subespacio. Evita construir
                # la proyección explícita, que sería una matriz D x D por clase.
                projected = centered @ basis.T
                residuals[start : start + block.shape[0], k] = np.einsum(
                    "ij,ij->i", centered, centered
                ) - np.einsum("ij,ij->i", projected, projected)

        # La resta puede dar un negativo diminuto por cancelación cuando el
        # espacio afín aproxima casi exactamente; geométricamente es un cero.
        return np.maximum(residuals, 0.0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmin(self._residuals(X), axis=1)]

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Puntuación creciente con la verosimilitud: el residuo cambiado de signo."""
        return -self._residuals(X)

    def approximation_errors(self, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """Métricas sigma_d^2 y lambda_d de la Tabla 5 del paper.

        `sigma_d2` es el error relativo de aproximación afín dentro de la clase
        correcta, promediado sobre clases. `lambda_d` es el cociente entre el
        error que produce la mejor clase equivocada y el de la clase correcta:
        cuanto mayor, más separadas están las clases en la representación.

        Sirven para diagnosticar *por qué* mejora el clasificador al crecer n,
        que es justo el análisis que el paper hace en su Tabla 5.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        residuals = self._residuals(X)

        sigma_terms = []
        ratio_terms = []

        for k, cls in enumerate(self.classes_):
            mask = y == cls
            if not np.any(mask):
                continue

            own = residuals[mask, k]
            energy = np.einsum("ij,ij->i", X[mask] - self.means_[k], X[mask] - self.means_[k])
            sigma_terms.append(np.mean(own) / np.mean(energy))

            others = np.delete(residuals[mask], k, axis=1)
            ratio_terms.append(np.mean(others.min(axis=1)) / np.mean(own))

        return float(np.mean(sigma_terms)), float(np.mean(ratio_terms))
