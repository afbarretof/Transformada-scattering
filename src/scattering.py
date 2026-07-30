"""Construcción de la transformada de scattering y etiquetado de sus caminos.

Kymatio 0.3.0 expone `meta()` solo para el scattering 1D; en 2D los canales de
salida vienen sin etiquetar. Como la interpretación de los resultados depende de
saber qué camino p = (j1, theta1, j2, theta2) produce cada canal, aquí se
reconstruye esa correspondencia replicando el orden de iteración de
`kymatio/scattering2d/core/scattering2d.py`, y se verifica contra la salida real.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from kymatio.torch import Scattering2D


@dataclass(frozen=True)
class Path:
    """Un camino de scattering p, es decir un canal de la salida.

    `order` es la longitud del camino: 0 para el promediado x * phi_J, 1 para
    |x * psi_{j1,theta1}| * phi_J, y 2 para el camino de dos wavelets.
    `j` y `theta` son las escalas y orientaciones recorridas, en orden.
    """

    order: int
    j: tuple[int, ...]
    theta: tuple[int, ...]

    def label(self) -> str:
        if self.order == 0:
            return "S0"
        pairs = ", ".join(f"({j},{t})" for j, t in zip(self.j, self.theta))
        return f"S{self.order}[{pairs}]"


def device_of(scattering: Scattering2D) -> torch.device:
    """Dispositivo en el que viven los filtros.

    No sirve mirar `parameters()`: el banco de filtros está registrado como
    buffers, no como parámetros, precisamente porque no se aprende nada. Un
    `Scattering2D` tiene 0 parámetros y 43 buffers, que es la comprobación más
    directa de que la representación no tiene nada que entrenar.
    """
    buffer = next(iter(scattering.buffers()), None)
    return buffer.device if buffer is not None else torch.device("cpu")


def num_learnable_parameters(scattering: Scattering2D) -> int:
    """Parámetros aprendibles del scattering: cero, por construcción."""
    return sum(p.numel() for p in scattering.parameters())


def build_scattering(
    J: int = 3,
    L: int = 8,
    shape: tuple[int, int] = (28, 28),
    max_order: int = 2,
    device: str | None = None,
) -> Scattering2D:
    """Instancia una Scattering2D, en GPU si hay una disponible.

    Los valores por defecto son los del experimento de MNIST de Bruna y Mallat:
    2^J = 8 (J = 3), elegido allí por validación cruzada.
    """
    scattering = Scattering2D(J=J, shape=shape, L=L, max_order=max_order)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return scattering.to(device)


def scattering_paths(scattering: Scattering2D) -> list[Path]:
    """Etiqueta cada canal de salida con el camino que lo genera.

    Reproduce el orden del core de Kymatio: primero el orden 0, luego los de
    orden 1 recorriendo el banco de filtros, y luego los de orden 2 con la
    restricción j2 > j1 (los caminos con j2 <= j1 tienen energía despreciable,
    porque el módulo de una wavelet ya es una señal de baja frecuencia).
    """
    psi = scattering.psi
    max_order = scattering.max_order

    paths: list[Path] = [Path(order=0, j=(), theta=())]

    for n1, p1 in enumerate(psi):
        paths.append(Path(order=1, j=(p1["j"],), theta=(p1["theta"],)))

    if max_order >= 2:
        for p1 in psi:
            for p2 in psi:
                if p2["j"] <= p1["j"]:
                    continue
                paths.append(
                    Path(
                        order=2,
                        j=(p1["j"], p2["j"]),
                        theta=(p1["theta"], p2["theta"]),
                    )
                )

    return paths


def verify_paths(scattering: Scattering2D) -> list[Path]:
    """Devuelve los caminos tras comprobarlos contra una pasada real.

    Barato y merece la pena: si una versión futura de Kymatio cambiara el orden
    de iteración, la correspondencia camino-canal se rompería en silencio y
    todas las figuras quedarían mal etiquetadas sin que nada fallase.
    """
    paths = scattering_paths(scattering)

    probe = torch.zeros(1, 1, *scattering.shape, device=device_of(scattering))
    n_channels = scattering(probe).shape[-3]

    if n_channels != len(paths):
        raise RuntimeError(
            f"Kymatio devolvió {n_channels} canales pero se etiquetaron {len(paths)} "
            "caminos; el orden de iteración del core ha cambiado."
        )
    return paths


def expected_num_paths(J: int, L: int, max_order: int = 2) -> int:
    """Número de caminos según la fórmula analítica, para contrastar.

    Orden 1 aporta J*L caminos y orden 2 aporta L^2 * C(J,2), ya que j2 > j1;
    más el único coeficiente de orden 0.
    """
    total = 1 + J * L
    if max_order >= 2:
        total += L**2 * (J * (J - 1)) // 2
    return total


def transform_dataset(
    images: np.ndarray,
    scattering: Scattering2D,
    batch_size: int = 256,
) -> np.ndarray:
    """Aplica el scattering a un array (N, H, W) y devuelve (N, canales, h, w).

    Se procesa por lotes porque la GTX 1050 Ti tiene 4 GB: el orden 2 con J=3
    genera 217 canales por imagen y un lote grande no cabe.
    """
    device = device_of(scattering)

    outputs = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            tensor = torch.from_numpy(batch).float().unsqueeze(1).to(device)
            outputs.append(scattering(tensor).squeeze(1).cpu().numpy())

    return np.concatenate(outputs, axis=0)
