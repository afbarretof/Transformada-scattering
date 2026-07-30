"""CNN de referencia, entrenada end-to-end.

El punto delicado de este módulo no es la arquitectura sino el **protocolo**. La
afirmación que sostiene todo el trabajo es que el scattering aventaja a una CNN
cuando hay pocos datos; esa afirmación no vale nada si la CNN está mal entrenada
a propósito o si recibe un presupuesto de datos distinto. Por eso:

1. La CNN nunca ve más de n muestras. El split de validación para la parada
   temprana sale *de dentro* de esas n, no de datos extra.

2. Tras elegir la época óptima sobre validación, se **reentrena desde cero con
   las n muestras completas** durante ese número de épocas. Sin este segundo
   paso la CNN entrenaría con 0.8n mientras la SVM y el PCA, vía el `refit` de
   GridSearchCV, se ajustan con las n completas: la comparación estaría sesgada
   en contra de la CNN, y precisamente a favor de nuestra hipótesis.

3. El mismo protocolo, sin excepciones, a cada tamaño n.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class SmallCNN(nn.Module):
    """CNN compacta para dígitos de 32x32.

    Dos bloques convolucionales con submuestreo y un clasificador lineal. La
    profundidad se elige para que la cascada de submuestreos sea análoga a la
    del scattering con J=3: ambas acaban con una rejilla espacial pequeña sobre
    la que se decide.
    """

    def __init__(self, n_classes: int = 10, width: int = 32, dropout: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, width, kernel_size=3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width, 2 * width, kernel_size=3, padding=1),
            nn.BatchNorm2d(2 * width),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * width, 2 * width, kernel_size=3, padding=1),
            nn.BatchNorm2d(2 * width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(2 * width * 16, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Número de parámetros del modelo."""
    parameters = model.parameters()
    if trainable_only:
        parameters = (p for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in parameters)


def select_best_epoch(val_curve: list[float], window: int = 3) -> int:
    """Época óptima según la curva de validación, suavizada primero.

    Con n pequeño el conjunto de validación es diminuto: para n=300 son 60
    muestras, así que el error de validación solo puede tomar valores múltiplos
    de 1/60 y su `argmin` crudo está dominado por el ruido de unas pocas
    imágenes. Tomar el mínimo de una media móvil corta escoge una *región*
    estable de la curva en lugar de un pico afortunado.

    Esto importa por equidad: un criterio ruidoso perjudicaría sistemáticamente
    a la CNN, que es justamente el modelo que la hipótesis del trabajo espera
    ver perder. Conviene que pierda, si pierde, por razones reales.

    En caso de empate se toma la época más tardía de la meseta: si dos regiones
    validan igual de bien, la más entrenada es la apuesta razonable.
    """
    if not val_curve:
        raise ValueError("La curva de validación está vacía.")

    curve = np.asarray(val_curve, dtype=float)
    window = int(min(window, len(curve)))
    kernel = np.ones(window) / window
    smoothed = np.convolve(curve, kernel, mode="valid")

    # `argmin` invertido para quedarnos con el último mínimo en caso de empate.
    best = len(smoothed) - 1 - int(np.argmin(smoothed[::-1]))

    # El índice de la media móvil apunta al inicio de la ventana; se devuelve su
    # centro, convertido a número de épocas (base 1).
    return int(best + (window + 1) // 2)


@dataclass
class TrainingOutcome:
    """Lo que devuelve un entrenamiento completo."""

    test_error: float
    best_epoch: int
    n_parameters: int
    seconds: float
    val_curve: list[float]


def _make_loader(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    tensors = TensorDataset(
        torch.from_numpy(images).float().unsqueeze(1),
        torch.from_numpy(labels).long(),
    )
    # generator fijo para que el barajado sea reproducible entre ejecuciones.
    generator = torch.Generator().manual_seed(0) if shuffle else None
    return DataLoader(
        tensors,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        drop_last=False,
    )


@torch.no_grad()
def _error_rate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    wrong = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        predictions = model(images).argmax(dim=1)
        wrong += int((predictions != labels).sum())
        total += len(labels)
    return wrong / total


def _train_epochs(
    model: nn.Module,
    loader: DataLoader,
    epochs: int,
    device: torch.device,
    learning_rate: float,
    weight_decay: float,
    val_loader: DataLoader | None = None,
) -> list[float]:
    """Entrena `epochs` épocas y devuelve la curva de error de validación."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    curve: list[float] = []
    for _ in range(epochs):
        model.train()
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        if val_loader is not None:
            curve.append(_error_rate(model, val_loader, device))

    return curve


def train_and_evaluate_cv(
    x_subset: np.ndarray,
    y_subset: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    n_folds: int = 3,
    max_epochs: int = 60,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-2,
    width: int = 32,
    batch_size: int | None = None,
    device: str | None = None,
) -> TrainingOutcome:
    """Elige la época por validación cruzada y reentrena con las n muestras.

    Sustituye al split único de validación, que a n pequeño no tiene señal: con
    n=300 aquel conjunto tenía 60 imágenes y su error oscilaba entre 3.33% y 5%,
    es decir entre dos y tres errores. Cambiar el criterio de selección sobre esa
    curva movía el error de test de 7.83% a 3.94%, lo que hacía que el resultado
    dependiese de una decisión que los datos no podían informar.

    Con k pliegues, cada muestra actúa como validación exactamente una vez, así
    que la curva promediada se estima sobre las n muestras en lugar de sobre
    0.2n. Además iguala el protocolo al de la SVM y el PCA, que ya elegían sus
    hiperparámetros por validación cruzada sobre el mismo presupuesto de datos.

    Coste: k+1 entrenamientos en vez de 2.
    """
    from sklearn.model_selection import StratifiedKFold

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    batch_size = batch_size or int(np.clip(len(x_subset) // 10, 8, 128))

    start = time.perf_counter()

    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_curves = []

    for fold_train, fold_val in splitter.split(x_subset, y_subset):
        torch.manual_seed(seed)
        model = SmallCNN(width=width).to(device)
        train_loader = _make_loader(
            x_subset[fold_train], y_subset[fold_train], batch_size, True, device
        )
        val_loader = _make_loader(x_subset[fold_val], y_subset[fold_val], 256, False, device)
        fold_curves.append(
            _train_epochs(
                model, train_loader, max_epochs, device, learning_rate, weight_decay, val_loader
            )
        )

    mean_curve = np.mean(fold_curves, axis=0).tolist()
    best_epoch = select_best_epoch(mean_curve)

    # Reentrenamiento final con las n muestras completas, igual que el refit de
    # GridSearchCV para los demás modelos.
    torch.manual_seed(seed)
    final_model = SmallCNN(width=width).to(device)
    full_loader = _make_loader(x_subset, y_subset, batch_size, True, device)
    _train_epochs(final_model, full_loader, best_epoch, device, learning_rate, weight_decay)

    test_loader = _make_loader(x_test, y_test, 512, False, device)
    error = _error_rate(final_model, test_loader, device)

    return TrainingOutcome(
        test_error=error,
        best_epoch=best_epoch,
        n_parameters=count_parameters(final_model),
        seconds=time.perf_counter() - start,
        val_curve=mean_curve,
    )


def train_and_evaluate(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    max_epochs: int = 60,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-2,
    width: int = 32,
    batch_size: int | None = None,
    device: str | None = None,
) -> TrainingOutcome:
    """Entrena la CNN en dos fases: buscar la época óptima, y reentrenar con todo.

    Devuelve el error sobre test del modelo reentrenado. La curva de validación
    de la primera fase se conserva para poder inspeccionar si el modelo llegó a
    sobreajustar, que es la comprobación de que la parada temprana hacía falta.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)

    # Con n pequeño, un lote grande deja poquísimos pasos de gradiente por
    # época; se limita a la décima parte del conjunto para que siga habiendo
    # suficientes actualizaciones.
    batch_size = batch_size or int(np.clip(len(x_train) // 10, 8, 128))

    start = time.perf_counter()

    # Fase 1: entrenar con el subconjunto y medir en validación cada época.
    torch.manual_seed(seed)
    model = SmallCNN(width=width).to(device)
    train_loader = _make_loader(x_train, y_train, batch_size, True, device)
    val_loader = _make_loader(x_val, y_val, 256, False, device)
    curve = _train_epochs(
        model, train_loader, max_epochs, device, learning_rate, weight_decay, val_loader
    )
    best_epoch = select_best_epoch(curve)

    # Fase 2: reentrenar desde cero con train + val durante `best_epoch` épocas,
    # para que la CNN disponga de las mismas n muestras que los demás modelos.
    full_x = np.concatenate([x_train, x_val])
    full_y = np.concatenate([y_train, y_val])

    torch.manual_seed(seed)
    final_model = SmallCNN(width=width).to(device)
    full_loader = _make_loader(full_x, full_y, batch_size, True, device)
    _train_epochs(final_model, full_loader, best_epoch, device, learning_rate, weight_decay)

    test_loader = _make_loader(x_test, y_test, 512, False, device)
    error = _error_rate(final_model, test_loader, device)

    return TrainingOutcome(
        test_error=error,
        best_epoch=best_epoch,
        n_parameters=count_parameters(final_model),
        seconds=time.perf_counter() - start,
        val_curve=curve,
    )
