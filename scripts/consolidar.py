"""Convierte los resultados crudos en la tabla y la figura del documento.

Uso:
    python scripts/consolidar.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")

from src.plotting import SERIES_LABELS, plot_data_curve, save_figure, use_paper_style

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "04_curva_datos.json"
REFERENCE = ROOT / "results" / "reference" / "bruna_mallat_2013_table4.csv"
TABLE_OUT = ROOT / "paper" / "tabla1.tex"

# Correspondencia entre nuestras claves de serie y las columnas del paper.
SERIES = {
    "pixels_svm": ("pixels", "svm", "pixels_svm"),
    "scat1_pca": ("scat1", "affine_pca", "scat1_pca"),
    "scat1_svm": ("scat1", "svm", "scat1_svm"),
    "scat2_pca": ("scat2", "affine_pca", "scat2_pca"),
    "scat2_svm": ("scat2", "svm", "scat2_svm"),
    "cnn": ("pixels", "cnn", "convnet"),
}

# Orden de filas en la tabla: de peor a mejor descriptor, con la CNN al final
# porque es el término de comparación.
ROW_ORDER = ["pixels_svm", "scat1_pca", "scat1_svm", "scat2_pca", "scat2_svm", "cnn"]


def aggregate(results: list[dict]) -> tuple[dict, dict, dict]:
    """Media, desviación típica y número de semillas por serie y tamaño."""
    buckets: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))

    for row in results:
        for key, (descriptor, model, _) in SERIES.items():
            if row["descriptor"] == descriptor and row["model"] == model:
                buckets[key][row["train_size"]].append(100 * row["error_rate"])

    means = {k: {n: float(np.mean(v)) for n, v in sizes.items()} for k, sizes in buckets.items()}
    deviations = {k: {n: float(np.std(v)) for n, v in sizes.items()} for k, sizes in buckets.items()}
    counts = {k: {n: len(v) for n, v in sizes.items()} for k, sizes in buckets.items()}
    return means, deviations, counts


def write_table(means: dict, deviations: dict, counts: dict, reference: pd.DataFrame) -> None:
    """Emite la tabla del artículo, a doble ancho de columna."""
    sizes = sorted({n for series in means.values() for n in series})

    lines = [
        "% Generado por scripts/consolidar.py — no editar a mano.",
        "\\begin{table*}[t]",
        "  \\centering",
        "  \\small",
        "  \\caption{Error de clasificación (\\%) en MNIST según el tamaño del",
        "  conjunto de entrenamiento. Media $\\pm$ desviación típica sobre cinco",
        "  semillas. Las dos últimas filas reproducen los valores publicados por",
        "  Bruna y Mallat para referencia.}",
        "  \\label{tab:resultados}",
        "  \\begin{tabular}{l" + "r" * len(sizes) + "}",
        "    \\toprule",
        "    Modelo & " + " & ".join(f"$n={n}$" for n in sizes) + " \\\\",
        "    \\midrule",
    ]

    for key in ROW_ORDER:
        if key not in means:
            continue
        cells = []
        for n in sizes:
            if n not in means[key]:
                cells.append("---")
                continue
            cell = f"{means[key][n]:.2f}"
            if counts[key][n] > 1:
                cell += f" $\\pm$ {deviations[key][n]:.2f}"
            cells.append(cell)
        lines.append(f"    {SERIES_LABELS[key]} & " + " & ".join(cells) + " \\\\")

    lines.append("    \\midrule")
    for column, label in (
        ("scat2_pca", "Bruna y Mallat, scattering orden 2 + PCA"),
        ("convnet", "Bruna y Mallat, ConvNet"),
    ):
        cells = []
        for n in sizes:
            match = reference.loc[reference["train_size"] == n, column]
            cells.append(f"{float(match.iloc[0]):.2f}" if len(match) else "---")
        lines.append(f"    \\emph{{{label}}} & " + " & ".join(cells) + " \\\\")

    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table*}",
        "",
    ]

    TABLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    TABLE_OUT.write_text("\n".join(lines), encoding="utf-8")


def summarise(means: dict, deviations: dict, counts: dict) -> None:
    """Resumen por pantalla, con el contraste que decide la tesis del trabajo."""
    sizes = sorted({n for series in means.values() for n in series})

    print("\nError medio (%) por modelo y tamaño:\n")
    frame = pd.DataFrame(
        {
            SERIES_LABELS[key]: {n: round(means[key].get(n, float("nan")), 2) for n in sizes}
            for key in ROW_ORDER
            if key in means
        }
    )
    print(frame.to_string())

    if "cnn" not in means or "scat2_pca" not in means:
        return

    print("\nScattering (orden 2 + PCA) frente a la CNN:\n")
    for n in sizes:
        if n not in means["cnn"] or n not in means["scat2_pca"]:
            continue
        scattering = means["scat2_pca"][n]
        cnn = means["cnn"][n]
        gap = cnn - scattering

        # Criterio deliberadamente conservador: se exige que la diferencia
        # supere la suma de las dispersiones, no solo que el signo sea el
        # esperado. Con cinco semillas y errores de décimas, cualquier cosa
        # menos estricta confundiría ruido con hallazgo.
        spread = deviations["scat2_pca"].get(n, 0) + deviations["cnn"].get(n, 0)
        if abs(gap) <= spread:
            verdict = "solapan"
        elif gap > 0:
            verdict = "gana scattering"
        else:
            verdict = "gana CNN"

        print(
            f"  n={n:6d}  scattering {scattering:5.2f}  cnn {cnn:5.2f}  "
            f"diferencia {gap:+5.2f} (±{spread:.2f})  ->  {verdict}"
        )

    incomplete = {
        (key, n)
        for key in means
        for n in means[key]
        if counts[key][n] < 5
    }
    if incomplete:
        print(f"\nAVISO: {len(incomplete)} configuraciones con menos de 5 semillas.")


def main() -> None:
    if not RESULTS.exists():
        raise SystemExit(f"No existe {RESULTS}. Ejecuta antes scripts/run_curva_datos.py.")

    results = json.loads(RESULTS.read_text(encoding="utf-8"))["results"]
    reference = pd.read_csv(REFERENCE, comment="#")

    means, deviations, counts = aggregate(results)

    use_paper_style()
    figure = plot_data_curve(
        means,
        deviations,
        series=["pixels_svm", "scat2_svm", "scat2_pca", "cnn"],
    )
    written = save_figure(figure, "fig08_curva_datos")

    write_table(means, deviations, counts, reference)

    summarise(means, deviations, counts)
    print(f"\nfigura: {', '.join(p.name for p in written)}")
    print(f"tabla:  {TABLE_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
