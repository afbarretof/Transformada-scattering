"""Utilidades de reproducibilidad.

Todo experimento del trabajo debe pasar por `set_seed` y registrar su entorno
con `environment_report`, de modo que cualquier número que aparezca en el
documento sea re-derivable.
"""

from __future__ import annotations

import importlib
import os
import platform
import random
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

# Librerías cuya versión queremos dejar registrada junto a cada resultado.
_TRACKED_PACKAGES = ("torch", "torchvision", "kymatio", "sklearn", "numpy", "scipy", "matplotlib")


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Fija las semillas de random, numpy y torch.

    Con `deterministic=True` además desactiva la autotunería de cuDNN. Eso
    cuesta algo de velocidad en la CNN, pero es lo que hace que la curva de
    precisión vs. tamaño de entrenamiento sea repetible entre ejecuciones.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@dataclass
class EnvironmentReport:
    """Instantánea del entorno, para adjuntar a los resultados guardados."""

    python: str
    platform: str
    packages: dict[str, str] = field(default_factory=dict)
    gpu: str | None = None
    git_commit: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def environment_report() -> EnvironmentReport:
    """Recoge versiones de librerías, GPU y commit actual."""
    packages: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        packages[name] = getattr(module, "__version__", "desconocida")

    gpu = None
    try:
        import torch

        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    return EnvironmentReport(
        python=platform.python_version(),
        platform=platform.platform(),
        packages=packages,
        gpu=gpu,
        git_commit=_git_commit(),
    )


def _git_commit() -> str | None:
    """Hash del commit actual, o None si no estamos en un repo git."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip()
