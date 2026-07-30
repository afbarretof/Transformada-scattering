"""Normalización de los coeficientes de scattering y caché en disco.

Bruna y Mallat usan **dos normalizaciones distintas**, una por clasificador, y
no es un detalle menor: la Tabla 1 del paper muestra que la energía se reparte
muy desigualmente entre caminos, sobre todo al cambiar de orden, así que sin
ecualizar el descriptor los caminos de orden 1 aplastan a los de orden 2.

- SVM: todos los coeficientes llevados a [-1, 1].
- PCA: ecuación (19), cada camino p dividido por el máximo sobre las señales de
  entrenamiento de ||S[p]X_i||, con la norma tomada sobre las posiciones u.

Ambas se ajustan **solo con el conjunto de entrenamiento** y luego se aplican al
test; hacerlo de otro modo filtraría información del test.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

CACHE_DIR = Path(__file__).resolve().parent.parent / "results" / "cache"


class PathNormalizer:
    """Ecuación (19): normaliza cada camino por su norma máxima en entrenamiento.

    Opera sobre arrays (N, canales, alto, ancho), es decir antes de aplanar,
    porque la norma de la ecuación (19) se toma sobre las posiciones espaciales
    de cada camino por separado.
    """

    def fit(self, features: np.ndarray) -> "PathNormalizer":
        # Norma L2 sobre las posiciones espaciales, por muestra y camino.
        norms = np.sqrt(np.sum(features**2, axis=(2, 3)))
        self.scale_ = norms.max(axis=0)
        # Un camino idénticamente nulo en entrenamiento no se puede normalizar;
        # se deja intacto en vez de dividir por cero.
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        return features / self.scale_[None, :, None, None]

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        return self.fit(features).transform(features)


def flatten(features: np.ndarray) -> np.ndarray:
    """Aplana (N, canales, alto, ancho) a (N, canales*alto*ancho)."""
    return features.reshape(features.shape[0], -1)


def select_orders(features: np.ndarray, paths: list, max_order: int) -> np.ndarray:
    """Se queda con los caminos de orden <= max_order.

    Permite obtener el descriptor de orden 1 a partir de los coeficientes ya
    calculados de orden 2, en vez de recomputar el scattering entero.
    """
    channels = [i for i, p in enumerate(paths) if p.order <= max_order]
    return features[:, channels]


def cache_key(**parameters) -> str:
    """Identificador estable de una configuración, para nombrar el caché."""
    payload = "|".join(f"{k}={parameters[k]}" for k in sorted(parameters))
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def cached_scattering(
    images: np.ndarray,
    scattering,
    tag: str,
    **key_parameters,
) -> np.ndarray:
    """Calcula el scattering, reutilizando el resultado en disco si existe.

    Recomputar todo MNIST cuesta unos 25 s en GPU, así que el caché no ahorra
    tanto tiempo como espacio mental: garantiza que todos los notebooks operan
    exactamente sobre los mismos números.
    """
    from src.scattering import transform_dataset

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{tag}_{cache_key(n=len(images), **key_parameters)}.npy"

    if path.exists():
        return np.load(path)

    features = transform_dataset(images, scattering)
    np.save(path, features)
    return features
