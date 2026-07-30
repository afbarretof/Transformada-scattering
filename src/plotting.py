"""Figuras del documento.

Convenciones de color, aplicadas según lo que representa cada campo:

- Wavelets en el espacio: son campos **con signo**, así que llevan un colormap
  divergente con gris neutro en el cero. Así el cero se lee como cero y no como
  "un color intermedio cualquiera".
- Módulos en Fourier y coeficientes de scattering: son **magnitudes** no
  negativas, así que llevan un secuencial de un solo tono, claro a oscuro.
- Nunca un arcoíris: introduce fronteras percibidas donde no hay saltos.

Todo se guarda en vectorial mediante
`save_figure`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"

# Divergente para campos con signo; secuencial de un tono para magnitudes.
CMAP_SIGNED = "RdBu_r"
CMAP_MAGNITUDE = "viridis"
CMAP_IMAGE = "gray_r"

PLOT_STYLE = {
    "figure.dpi": 110,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def use_paper_style() -> None:
    """Estilo común a todas las figuras del documento."""
    plt.rcParams.update(PLOT_STYLE)


def save_figure(fig: plt.Figure, name: str, formats: tuple[str, ...] = ("pdf", "svg")) -> list[Path]:
    """Guarda una figura en formato vectorial y devuelve las rutas escritas."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = FIGURES_DIR / f"{name}.{fmt}"
        fig.savefig(path, format=fmt)
        paths.append(path)
    return paths


def _hide_ticks(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_filter_bank_spatial(
    filters: dict,
    J: int,
    L: int,
    part: str = "real",
) -> plt.Figure:
    """Banco de wavelets de Morlet en el dominio espacial, una fila por escala.

    Kymatio guarda los filtros en Fourier, así que hay que antitransformar. El
    `fftshift` recentra la wavelet: sin él aparece partida por las cuatro
    esquinas, que es dónde vive el origen en la convención de la FFT.

    `part` selecciona parte real o imaginaria. Son la misma wavelet desfasada
    90 grados: la real es par y la imaginaria impar, y es justamente esa pareja
    en cuadratura la que hace que el módulo sea una envolvente suave y estable.
    """
    fig, axes = plt.subplots(J, L, figsize=(L * 1.05, J * 1.15))
    axes = np.atleast_2d(axes)

    for psi in filters["psi"]:
        j, theta = psi["j"], psi["theta"]
        spatial = np.fft.fftshift(np.fft.ifft2(psi["levels"][0]))
        field = spatial.real if part == "real" else spatial.imag

        ax = axes[j, theta]
        limit = np.abs(field).max()
        ax.imshow(field, cmap=CMAP_SIGNED, vmin=-limit, vmax=limit)
        _hide_ticks(ax)
        if theta == 0:
            ax.set_ylabel(f"$j={j}$", rotation=0, ha="right", va="center", labelpad=12)
        if j == 0:
            ax.set_title(rf"$\theta_{{{theta}}}$", pad=4)

    label = "real" if part == "real" else "imaginaria"
    fig.suptitle(f"Banco de wavelets de Morlet (parte {label}), $J={J}$, $L={L}$", y=1.0)
    fig.tight_layout()
    return fig


def plot_filter_bank_fourier(filters: dict, J: int, L: int) -> plt.Figure:
    """Los mismos filtros en Fourier, más el paso bajo phi.

    Esta es la figura que explica la construcción: cada wavelet es un bulto
    localizado en frecuencia, las orientaciones cubren el plano en abanico, y
    al bajar de escala los bultos se acercan al origen y se estrechan. El hueco
    que queda en el centro es exactamente lo que tapa phi, y es la razón de que
    haga falta el coeficiente de orden 0.
    """
    fig, axes = plt.subplots(J, L + 1, figsize=((L + 1) * 1.05, J * 1.15))
    axes = np.atleast_2d(axes)

    for psi in filters["psi"]:
        j, theta = psi["j"], psi["theta"]
        modulus = np.fft.fftshift(np.abs(psi["levels"][0]))
        ax = axes[j, theta]
        ax.imshow(modulus, cmap=CMAP_MAGNITUDE)
        _hide_ticks(ax)
        if theta == 0:
            ax.set_ylabel(f"$j={j}$", rotation=0, ha="right", va="center", labelpad=12)
        if j == 0:
            ax.set_title(rf"$\theta_{{{theta}}}$", pad=4)

    # Última columna: el paso bajo, idéntico en todas las filas. Se dibuja una
    # sola vez y las demás celdas se apagan, para no sugerir que varía con j.
    phi_modulus = np.fft.fftshift(np.abs(filters["phi"]["levels"][0]))
    axes[0, L].imshow(phi_modulus, cmap=CMAP_MAGNITUDE)
    axes[0, L].set_title(r"$\phi_J$", pad=4)
    _hide_ticks(axes[0, L])
    for j in range(1, J):
        axes[j, L].axis("off")

    fig.suptitle(rf"Filtros en el plano de Fourier $|\hat\psi_{{j,\theta}}|$ y $|\hat\phi_J|$, $J={J}$, $L={L}$", y=1.0)
    fig.tight_layout()
    return fig


def plot_scattering_coefficients(
    image: np.ndarray,
    coefficients: np.ndarray,
    paths: list,
    L: int,
    max_display: int = 24,
) -> plt.Figure:
    """La imagen original junto a sus coeficientes de orden 0, 1 y 2.

    Los de orden 1 se ordenan por (j, theta) para que se lea el barrido de
    escalas y orientaciones. Los de orden 2 son demasiados para mostrarlos
    todos (192 con J=3, L=8), así que se muestran los de mayor energía, que son
    los que de verdad pesan en la clasificación.
    """
    order1 = [i for i, p in enumerate(paths) if p.order == 1]
    order2 = [i for i, p in enumerate(paths) if p.order == 2]

    energies = np.array([np.sum(coefficients[i] ** 2) for i in order2])
    top_order2 = [order2[k] for k in np.argsort(energies)[::-1][:max_display]]

    n_cols = L
    rows_1 = int(np.ceil(len(order1) / n_cols))
    rows_2 = int(np.ceil(len(top_order2) / n_cols))
    total_rows = 1 + rows_1 + rows_2

    fig, axes = plt.subplots(total_rows, n_cols, figsize=(n_cols * 1.05, total_rows * 1.2))
    for ax in axes.ravel():
        ax.axis("off")

    axes[0, 0].imshow(image, cmap=CMAP_IMAGE)
    axes[0, 0].set_title("$x$", pad=4)
    axes[0, 0].axis("on")
    _hide_ticks(axes[0, 0])

    idx0 = next(i for i, p in enumerate(paths) if p.order == 0)
    axes[0, 1].imshow(coefficients[idx0], cmap=CMAP_MAGNITUDE)
    axes[0, 1].set_title(r"$S_0 = x * \phi_J$", pad=4)
    axes[0, 1].axis("on")
    _hide_ticks(axes[0, 1])

    for k, channel in enumerate(order1):
        ax = axes[1 + k // n_cols, k % n_cols]
        ax.imshow(coefficients[channel], cmap=CMAP_MAGNITUDE)
        ax.axis("on")
        _hide_ticks(ax)
        p = paths[channel]
        ax.set_title(rf"$j={p.j[0]},\ \theta={p.theta[0]}$", pad=3, fontsize=7)

    for k, channel in enumerate(top_order2):
        ax = axes[1 + rows_1 + k // n_cols, k % n_cols]
        ax.imshow(coefficients[channel], cmap=CMAP_MAGNITUDE)
        ax.axis("on")
        _hide_ticks(ax)
        p = paths[channel]
        ax.set_title(
            rf"$({p.j[0]},{p.theta[0]})\!\to\!({p.j[1]},{p.theta[1]})$",
            pad=3,
            fontsize=6,
        )

    fig.suptitle(
        "Coeficientes de scattering: orden 0 y 1 (arriba), "
        f"y los {len(top_order2)} de orden 2 con más energía (abajo)",
        y=1.0,
    )
    fig.tight_layout()
    return fig


# Paleta categórica de orden fijo: cada modelo conserva su color aunque cambie
# el conjunto de modelos dibujados. Validada para separación bajo daltonismo.
SERIES_COLORS = {
    "pixels_svm": "#2a78d6",
    "scat1_pca": "#008300",
    "scat2_pca": "#e87ba4",
    "scat2_svm": "#eda100",
    "cnn": "#4a3aa7",
}

# Hay más etiquetas que colores a propósito: la tabla del artículo lista los seis
# modelos, mientras que la figura solo traza cuatro. Añadir un color por serie
# tabulada obligaría a ampliar la paleta más allá de lo que se ha validado para
# separación bajo daltonismo, sin que ninguna figura lo use.
SERIES_LABELS = {
    "pixels_svm": "píxeles + SVM",
    "scat1_pca": "scattering orden 1 + PCA",
    "scat1_svm": "scattering orden 1 + SVM",
    "scat2_pca": "scattering orden 2 + PCA",
    "scat2_svm": "scattering orden 2 + SVM",
    "cnn": "CNN entrenada",
}


def _stagger_offsets(
    values: list[float],
    min_ratio: float = 0.08,
    step: int = 30,
    base: int = 8,
) -> list[int]:
    """Desplazamientos **horizontales** para etiquetas que se pisarían.

    Se separan en horizontal y no en vertical porque a la derecha de la última
    marca el espacio está libre, mientras que desplazar hacia arriba solo mueve
    el problema a la etiqueta siguiente: al converger tres curvas, empujar la
    de abajo la mete encima de la de en medio.

    La proximidad se mide en términos *relativos* porque el eje es logarítmico:
    una separación de 0.04 es ilegible cerca de 0.8 y holgada cerca de 10.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    offsets = [base] * len(values)

    for rank, index in enumerate(order):
        if rank == 0:
            continue
        previous = values[order[rank - 1]]
        if previous <= 0:
            continue
        if (values[index] - previous) / previous < min_ratio:
            offsets[index] = offsets[order[rank - 1]] + step

    return offsets


def plot_data_curve(
    means: dict[str, dict[int, float]],
    deviations: dict[str, dict[int, float]],
    series: list[str] | None = None,
) -> plt.Figure:
    """Figura central: error frente al tamaño de entrenamiento, con dispersión.

    Las barras cubren una desviación típica sobre las semillas. Es la
    información que decide si una diferencia entre modelos significa algo: a
    n pequeño la varianza entre submuestras es grande, y sin ella dos curvas
    que se cruzan podrían estar simplemente solapadas.

    Ambos ejes en logarítmico, para que una misma reducción relativa del error
    ocupe la misma distancia vertical en todo el rango.
    """
    series = series or list(SERIES_COLORS)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    endpoints: list[tuple[float, float]] = []

    for key in series:
        # Se omiten las series sin datos y las que no tienen color asignado:
        # estas últimas existen para la tabla, no para la figura.
        if key not in means or key not in SERIES_COLORS:
            continue
        sizes = sorted(means[key])
        values = np.array([means[key][s] for s in sizes])
        spread = np.array([deviations.get(key, {}).get(s, 0.0) for s in sizes])

        ax.errorbar(
            sizes,
            values,
            yerr=spread,
            color=SERIES_COLORS[key],
            linewidth=2.0,
            marker="o",
            markersize=5.5,
            markeredgecolor="white",
            markeredgewidth=1.0,
            capsize=3,
            elinewidth=1.2,
            label=SERIES_LABELS[key],
            zorder=3,
        )
        endpoints.append((sizes[-1], values[-1]))

    # Las etiquetas finales se separan verticalmente cuando dos curvas terminan
    # casi juntas, que es justo lo que ocurre cuando los modelos convergen: sin
    # esto, el punto más interesante de la figura es el que queda ilegible.
    for (x, y), offset in zip(endpoints, _stagger_offsets([y for _, y in endpoints])):
        ax.annotate(
            f"{y:.2f}",
            (x, y),
            textcoords="offset points",
            xytext=(offset, 0),
            fontsize=7.5,
            color="#52514e",
            va="center",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("tamaño del conjunto de entrenamiento ($n$)")
    ax.set_ylabel("error de clasificación (%)")
    ax.grid(alpha=0.22, linewidth=0.6, which="both")
    ax.set_axisbelow(True)
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")

    fig.tight_layout()
    return fig


def plot_replication_comparison(
    ours: dict[str, dict[int, float]],
    reference: dict[str, dict[int, float]],
) -> plt.Figure:
    """Nuestra réplica frente a la Tabla 4 del paper.

    `ours` y `reference` son {serie: {tamaño: error en %}}. Trazo continuo para
    nuestras cifras y discontinuo para las del paper: la distinción no descansa
    solo en el color, así que sigue leyéndose impresa en blanco y negro.

    Ambos ejes en escala logarítmica. El tamaño de entrenamiento porque recorre
    dos órdenes de magnitud, y el error porque lo que interesa comparar son
    reducciones relativas: pasar de 5% a 2.5% y de 1% a 0.5% son la misma mejora
    y deben ocupar la misma distancia vertical.
    """
    fig, ax = plt.subplots(figsize=(6.0, 4.2))

    for key, color in SERIES_COLORS.items():
        if key in reference:
            sizes = sorted(reference[key])
            ax.plot(
                sizes,
                [reference[key][s] for s in sizes],
                linestyle="--",
                linewidth=1.4,
                color=color,
                alpha=0.55,
                zorder=2,
            )
        if key in ours:
            sizes = sorted(ours[key])
            values = [ours[key][s] for s in sizes]
            ax.plot(
                sizes,
                values,
                linestyle="-",
                linewidth=2.0,
                marker="o",
                markersize=5.5,
                markeredgecolor="white",
                markeredgewidth=1.0,
                color=color,
                zorder=3,
                label=SERIES_LABELS[key],
            )
            # Etiqueta directa junto al último punto: el contraste de algunos
            # tonos sobre blanco es bajo, así que la identidad no puede
            # descansar solo en el color.
            ax.annotate(
                f"{values[-1]:.2f}%",
                (sizes[-1], values[-1]),
                textcoords="offset points",
                xytext=(7, 0),
                fontsize=7.5,
                color="#52514e",
                va="center",
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("tamaño del conjunto de entrenamiento")
    ax.set_ylabel("error de clasificación (%)")
    ax.grid(alpha=0.22, linewidth=0.6, which="both")
    ax.set_axisbelow(True)

    from matplotlib.lines import Line2D

    handles, labels = ax.get_legend_handles_labels()
    handles += [
        Line2D([], [], color="#52514e", linestyle="-", linewidth=2.0),
        Line2D([], [], color="#52514e", linestyle="--", linewidth=1.4, alpha=0.55),
    ]
    labels += ["esta réplica", "Bruna y Mallat (2013)"]
    ax.legend(handles, labels, fontsize=7.5, frameon=False, loc="lower left")

    fig.tight_layout()
    return fig


def plot_validation_curves(
    curves: dict[int, list[float]],
    chosen_epochs: dict[int, int],
) -> plt.Figure:
    """Curvas de error de validación de la CNN, una por tamaño de entrenamiento.

    Documenta el protocolo de parada temprana. Lo que hay que mirar no es solo
    dónde está el mínimo, sino **cuánto ruido tiene la curva**: con n pequeño el
    conjunto de validación es diminuto y su error se mueve a saltos de una
    imagen, de ahí que la época se elija sobre la curva suavizada.
    """
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    palette = ["#2a78d6", "#008300", "#e87ba4", "#eda100"]

    for color, (size, curve) in zip(palette, sorted(curves.items())):
        epochs = np.arange(1, len(curve) + 1)
        values = 100 * np.asarray(curve)
        ax.plot(epochs, values, color=color, linewidth=1.6, label=f"n = {size}")

        chosen = chosen_epochs.get(size)
        if chosen is not None and 1 <= chosen <= len(values):
            ax.plot(
                chosen,
                values[chosen - 1],
                marker="o",
                markersize=8,
                color=color,
                markeredgecolor="white",
                markeredgewidth=1.4,
                zorder=4,
            )

    ax.set_xlabel("época")
    ax.set_ylabel("error de validación (%)")
    ax.set_yscale("log")
    ax.grid(alpha=0.22, linewidth=0.6, which="both")
    ax.set_axisbelow(True)

    from matplotlib.lines import Line2D

    handles, labels = ax.get_legend_handles_labels()
    handles.append(
        Line2D([], [], marker="o", markersize=8, linestyle="none", color="#52514e")
    )
    labels.append("época elegida")
    ax.legend(handles, labels, fontsize=7.5, frameon=False)

    fig.tight_layout()
    return fig


def plot_energy_by_order(coefficients: np.ndarray, paths: list) -> plt.Figure:
    """Energía media por orden: por qué el orden 3 no compensa.

    Bruna y Mallat argumentan que los caminos de longitud 3 tienen energía
    despreciable y aportan poco al clasificador. Esta figura lo comprueba sobre
    nuestros propios datos en vez de citarlo, que es la diferencia entre
    replicar y parafrasear.
    """
    orders = sorted({p.order for p in paths})
    energies = []
    for order in orders:
        channels = [i for i, p in enumerate(paths) if p.order == order]
        energies.append(float(np.sum(coefficients[channels] ** 2)))

    total = sum(energies)
    fractions = [100 * e / total for e in energies]

    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    bars = ax.bar([f"orden {o}" for o in orders], fractions, width=0.55, color="#4C72B0")
    for bar, fraction in zip(bars, fractions):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{fraction:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylabel("porcentaje de la energía total")
    ax.set_ylim(0, max(fractions) * 1.18)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig
