"""Precisión frente al tamaño del conjunto de entrenamiento.

Recorre tamaños y semillas, evaluando los cuatro modelos con protocolos
equivalentes, todos ven exactamente las mismas n muestras y todos eligen sus
hiperparámetros por validación cruzada dentro de ellas, sin mirar el test.

Los resultados se van escribiendo a disco después de cada configuración, de modo
que una interrupción no obliga a repetir lo ya hecho y una segunda ejecución
reanuda donde se quedó.

Uso:
    python scripts/run_curva_datos.py                # todo
    python scripts/run_curva_datos.py --sizes 300 1000 --seeds 0 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cnn import train_and_evaluate_cv
from src.data import load_mnist, pad_to_square, stratified_subsample
from src.evaluation import RESULTS_DIR, Result, evaluate_affine_pca, evaluate_svm
from src.features import PathNormalizer, cached_scattering, flatten, select_orders
from src.repro import environment_report, set_seed
from src.scattering import build_scattering, verify_paths

J, L, SIZE = 3, 8, 32
DEFAULT_SIZES = [300, 1000, 2000, 5000, 10000]
DEFAULT_SEEDS = [0, 1, 2, 3, 4]
OUTPUT = RESULTS_DIR / "04_curva_datos.json"


def load_done() -> tuple[list[dict], set[tuple]]:
    """Resultados ya calculados y sus claves, para poder reanudar."""
    if not OUTPUT.exists():
        return [], set()
    stored = json.loads(OUTPUT.read_text(encoding="utf-8"))["results"]
    keys = {(r["model"], r["descriptor"], r["train_size"], r["seed"]) for r in stored}
    return stored, keys


def flush(results: list[dict]) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {"environment": environment_report().to_dict(), "results": results}
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    args = parser.parse_args()

    results, done = load_done()
    print(f"reanudando con {len(done)} configuraciones ya hechas", flush=True)

    x_train, y_train, x_test, y_test = load_mnist()
    train_padded = pad_to_square(x_train, SIZE)
    test_padded = pad_to_square(x_test, SIZE)
    pixels_train = x_train.reshape(len(x_train), -1)
    pixels_test = x_test.reshape(len(x_test), -1)

    scattering = build_scattering(J=J, L=L, shape=(SIZE, SIZE), max_order=2)
    paths = verify_paths(scattering)
    features_train = cached_scattering(
        train_padded, scattering, "mnist_train", J=J, L=L, size=SIZE, order=2
    )
    features_test = cached_scattering(
        test_padded, scattering, "mnist_test", J=J, L=L, size=SIZE, order=2
    )

    for seed in args.seeds:
        for n in args.sizes:
            set_seed(seed)
            subset = stratified_subsample(y_train, n, seed=seed)
            labels = y_train[subset]

            def record(result: Result) -> None:
                results.append(result.to_dict())
                flush(results)
                print(
                    f"  [{result.model:>10} {result.descriptor:>7}] "
                    f"error={result.error_rate*100:5.2f}%  ({result.fit_seconds:.0f}s)",
                    flush=True,
                )

            print(f"n={n} semilla={seed}", flush=True)

            if ("svm", "pixels", n, seed) not in done:
                record(
                    evaluate_svm(
                        pixels_train[subset], labels, pixels_test, y_test, "pixels", seed
                    )
                )

            for order in (1, 2):
                descriptor = f"scat{order}"
                train_sub = select_orders(features_train[subset], paths, order)
                test_sub = select_orders(features_test, paths, order)

                if ("affine_pca", descriptor, n, seed) not in done:
                    normalizer = PathNormalizer().fit(train_sub)
                    record(
                        evaluate_affine_pca(
                            flatten(normalizer.transform(train_sub)),
                            labels,
                            flatten(normalizer.transform(test_sub)),
                            y_test,
                            descriptor,
                            seed,
                        )
                    )

                if ("svm", descriptor, n, seed) not in done:
                    record(
                        evaluate_svm(
                            flatten(train_sub), labels, flatten(test_sub), y_test, descriptor, seed
                        )
                    )

            if ("cnn", "pixels", n, seed) not in done:
                outcome = train_and_evaluate_cv(
                    train_padded[subset], labels, test_padded, y_test, seed=seed
                )
                record(
                    Result(
                        model="cnn",
                        descriptor="pixels",
                        train_size=n,
                        seed=seed,
                        error_rate=outcome.test_error,
                        best_params={"best_epoch": str(outcome.best_epoch)},
                        n_features=SIZE * SIZE,
                        fit_seconds=outcome.seconds,
                        extra={"n_parameters": outcome.n_parameters},
                    )
                )

    print(f"\nlisto: {len(results)} resultados en {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
