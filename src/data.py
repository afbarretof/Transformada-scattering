"""Carga de MNIST y construcción de submuestras para el experimento central.

El punto delicado de la curva "precisión vs. tamaño de entrenamiento" es que las
submuestras sean honestas: estratificadas por clase, anidadas al crecer n, y con
un split de validación que nunca toque el test. Todo eso vive aquí para que los
cuatro modelos (píxeles+SVM, scattering+SVM, PCA generativo, CNN) reciban
exactamente los mismos índices.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_mnist(root: Path | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve (x_train, y_train, x_test, y_test) de MNIST.

    Las imágenes salen como float32 en [0, 1] con forma (N, 28, 28); Kymatio y
    la CNN añaden después el eje de canal según convenga.
    """
    from torchvision import datasets

    root = root or DATA_DIR
    root.mkdir(parents=True, exist_ok=True)

    train = datasets.MNIST(root=str(root), train=True, download=True)
    test = datasets.MNIST(root=str(root), train=False, download=True)

    x_train = train.data.numpy().astype(np.float32) / 255.0
    y_train = train.targets.numpy().astype(np.int64)
    x_test = test.data.numpy().astype(np.float32) / 255.0
    y_test = test.targets.numpy().astype(np.int64)

    return x_train, y_train, x_test, y_test


def pad_to_square(images: np.ndarray, size: int = 32) -> np.ndarray:
    """Centra imágenes de 28x28 en un lienzo de 32x32.

    No es cosmético. Con 2^J = 8, un lienzo de 32 deja una rejilla espacial de
    32/8 = 4 puntos por lado, que es exactamente el array 4x4 de la Figura 7 de
    Bruna y Mallat; sobre los 28 píxeles crudos saldría 3x3 y las figuras no
    serían comparables con las suyas. Además 32 es potencia de 2, que es el
    caso limpio para la cascada de submuestreos.
    """
    n, height, width = images.shape
    if height > size or width > size:
        raise ValueError(f"Las imágenes ({height}x{width}) no caben en {size}x{size}.")

    top = (size - height) // 2
    left = (size - width) // 2
    canvas = np.zeros((n, size, size), dtype=images.dtype)
    canvas[:, top : top + height, left : left + width] = images
    return canvas


def stratified_subsample(
    y: np.ndarray,
    n_total: int,
    seed: int,
) -> np.ndarray:
    """Índices de una submuestra estratificada de tamaño `n_total`.

    Se reparte n_total entre las clases lo más uniformemente posible. Con los
    tamaños pequeños que nos interesan (n = 100, 300, 1000...) dejar que el
    muestreo sea puramente aleatorio produce clases con muy pocos ejemplos y
    mete varianza que no tiene nada que ver con el fenómeno que estudiamos.
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    base, extra = divmod(n_total, len(classes))

    # Las clases que reciben un ejemplo de más se eligen al azar, para no
    # favorecer sistemáticamente a las de índice bajo.
    bonus = set(rng.choice(classes, size=extra, replace=False).tolist())

    indices: list[np.ndarray] = []
    for cls in classes:
        pool = np.flatnonzero(y == cls)
        take = base + (1 if cls in bonus else 0)
        if take > len(pool):
            raise ValueError(f"La clase {cls} solo tiene {len(pool)} ejemplos, se pidieron {take}.")
        indices.append(rng.choice(pool, size=take, replace=False))

    selected = np.concatenate(indices)
    rng.shuffle(selected)
    return selected


def train_val_split(
    y: np.ndarray,
    indices: np.ndarray,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Parte un conjunto de índices en (train, val) de forma estratificada.

    La CNN necesita validación para early stopping y la SVM para elegir sus
    hiperparámetros. Que ambos la saquen de aquí, del mismo presupuesto de n
    ejemplos, es lo que hace justa la comparación: ningún modelo ve más datos
    que otro.
    """
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_fraction,
        random_state=seed,
        stratify=y[indices],
    )
    return train_idx, val_idx
