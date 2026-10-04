"""Prepare les donnees de l'interface web (`ui/`).

Usage (depuis la racine du projet) :
    python scripts/export_ui.py            # tout (environ 10 minutes)
    python scripts/export_ui.py --no-signal  # sans re-exporter les signaux

Produit dans `ui/data/` :
    meta.json            enregistrements, crises, fenetres, caracteristiques,
                         probabilites des modeles, resultats, arbres
    sig/<id>/<k>.png     signal filtre, 19 canaux, 128 Hz, int16 (0,1 µV),
                         par blocs de 300 s (charges a la demande par l'interface).
                         Les entiers sont ranges dans une image PNG en niveaux de
                         gris (format compresse sans perte, servi partout) : pour
                         chaque voie, differences successives, octets de poids
                         fort puis octets de poids faible. Voir `write_png`.

Les probabilites affichees sur le trace sont obtenues **par patient** :
chaque enregistrement est predit par un modele entraine sans lui
(GroupKFold, 5 plis, hyperparametres de la Question 12). C'est la situation
reelle d'un nouveau patient.
"""

from __future__ import annotations

import argparse
import json
import struct
import zlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.base import clone  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from src.data import annotation_issues, file_id, load_recording, load_seizure_annotations  # noqa: E402
from src.evaluation import compute_metrics, confusion_table, results_table  # noqa: E402
from src.exploration import class_comparison, seizure_statistics  # noqa: E402
from src.features import BANDS, EEG_CHANNELS, FEATURES, build_dataset  # noqa: E402
from src.preprocessing import clean_recording  # noqa: E402
from src.training import depth_curve, split_data, tune_models  # noqa: E402

OUT = ROOT / "ui" / "data"
UI_RATE = 128           # Hz : le signal est deja filtre sous 40 Hz, 128 Hz suffit
DECIMATE = 2            # 256 Hz -> 128 Hz
AMP_STEP = 0.1          # µV par unite int16 (plage +- 3 276 µV)
CHUNK_S = 300           # duree d'un bloc de signal


def sig(x, digits=4):
    """Arrondi a `digits` chiffres significatifs (JSON plus leger)."""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    nz = np.isfinite(x) & (x != 0)
    mag = np.floor(np.log10(np.abs(x[nz])))
    factor = 10.0 ** (digits - 1 - mag)
    out[nz] = np.round(x[nz] * factor) / factor
    return [float(v) for v in out]


def write_png(path, payload: bytes, width: int = 4096):
    """Ecrit des octets bruts dans un PNG gris 8 bits (filtre 0, sans perte)."""
    height = -(-len(payload) // width)
    payload = payload + bytes(width * height - len(payload))
    raw = b"".join(b"\x00" + payload[r * width:(r + 1) * width] for r in range(height))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    path.write_bytes(png)


def encode_block(block: np.ndarray) -> bytes:
    """(voies, n) int16 -> differences successives, plan fort puis plan faible par voie."""
    delta = np.diff(block, axis=1, prepend=np.zeros((block.shape[0], 1), dtype=block.dtype))
    u = delta.astype("<i2").view("<u2")
    parts = []
    for row in u:
        parts.append((row >> 8).astype(np.uint8).tobytes())
        parts.append((row & 0xFF).astype(np.uint8).tobytes())
    return b"".join(parts)


def export_signals(recording_ids):
    for rid in recording_ids:
        t0 = time.time()
        rec = load_recording(rid, ROOT, with_events=False)
        clean = clean_recording(rec, channels=EEG_CHANNELS)
        x = clean.data[EEG_CHANNELS].to_numpy()[::DECIMATE]          # (n, 19)
        q = np.clip(np.round(x / AMP_STEP), -32767, 32767).astype("<i2")
        folder = OUT / "sig" / rid
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.glob("*.bin"):
            old.unlink()
        per_chunk = CHUNK_S * UI_RATE
        for k, start in enumerate(range(0, len(q), per_chunk)):
            block = np.ascontiguousarray(q[start : start + per_chunk].T)  # voie par voie
            write_png(folder / f"{k}.png", encode_block(block))
        print(f"  signal {rid}: {len(q)} echantillons, {k + 1} blocs ({time.time() - t0:.1f} s)")


def no_nan(obj):
    """JSON n'accepte pas NaN : on le remplace par null."""
    if isinstance(obj, dict):
        return {k: no_nan(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [no_nan(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, np.generic):
        return no_nan(obj.item())
    return obj


def tree_json(tree):
    t = tree.tree_
    counts = t.value[:, 0, :] * t.n_node_samples[:, None]  # sklearn >= 1.4 : proportions
    return {
        "left": t.children_left.tolist(),
        "right": t.children_right.tolist(),
        "feature": [FEATURES[f] if f >= 0 else None for f in t.feature],
        "threshold": sig(np.where(t.feature >= 0, t.threshold, 0), 5),
        "impurity": sig(t.impurity, 4),
        "samples": t.n_node_samples.tolist(),
        "value": np.round(counts).astype(int).tolist(),
        "depth": int(tree.get_depth()),
        "leaves": int(tree.get_n_leaves()),
    }


def main(with_signal=True):
    OUT.mkdir(parents=True, exist_ok=True)
    annotations = load_seizure_annotations(ROOT)
    dataset = build_dataset(annotations, ROOT, cache=ROOT / "data/processed/windows_features.csv")
    recording_ids = list(dict.fromkeys(dataset["recording_id"]))

    if with_signal:
        print("Signaux")
        export_signals(recording_ids)

    # ---------------------------------------------------------------- modeles
    print("Recherche d'hyperparametres (comme la Question 12)")
    X_train, X_test, y_train, y_test = split_data(dataset, FEATURES)
    tuned = tune_models(X_train, y_train)
    names = [n for n in tuned if not n.startswith("Reference")]

    test_table = results_table(tuned, X_test, y_test)
    confusions = confusion_table(tuned, X_test, y_test)

    print("Probabilites par patient (GroupKFold)")
    X, y, groups = dataset[FEATURES], dataset["Class"], dataset["recording_id"]
    oof = {name: np.zeros(len(dataset)) for name in names}
    folds = list(GroupKFold(n_splits=5).split(X, y, groups))
    fold_of = np.zeros(len(dataset), dtype=int)
    for k, (_, te) in enumerate(folds):
        fold_of[te] = k
    for name in names:
        t0 = time.time()
        base = tuned[name].estimator
        for tr, te in folds:
            model = clone(base).fit(X.iloc[tr], y.iloc[tr])
            if name.startswith("SVM"):
                # Le SVM donne une distance a la frontiere, pas une probabilite :
                # on la passe dans une sigmoide, donc score >= 0,5 <=> prediction "crise".
                oof[name][te] = 1.0 / (1.0 + np.exp(-model.decision_function(X.iloc[te])))
            else:
                oof[name][te] = model.predict_proba(X.iloc[te])[:, 1]
        print(f"  {name}: {time.time() - t0:.1f} s")

    group_metrics = {
        name: compute_metrics(y, (oof[name] >= 0.5).astype(int)) for name in names
    }

    # leaf de chaque fenetre dans les deux arbres (pour surligner le chemin)
    leaves = {
        key: tuned[name].estimator.apply(X).tolist()
        for key, name in (("gini", "Decision Tree (Gini)"), ("entropy", "Decision Tree (Entropie)"))
    }
    in_test = np.zeros(len(dataset), dtype=int)
    in_test[X_test.index.to_numpy()] = 1

    curve = depth_curve(X_train, y_train)
    comparison = class_comparison(dataset, FEATURES)
    corr = dataset[FEATURES].corr()
    stats = seizure_statistics(annotations)
    rf = tuned["Random Forest"].estimator

    # ------------------------------------------------------------ assemblage
    recordings = []
    windows = {}
    for rid in recording_ids:
        part = dataset[dataset["recording_id"] == rid]
        idx = part.index.to_numpy()
        excel = rid.rsplit("_", 1)[0].replace("_", "-")
        crises = annotations[annotations["recording_id"] == excel].sort_values("start_s")
        n_samples = int((len(part) + 1) * UI_RATE)
        recordings.append({
            "id": rid,
            "label": excel,
            "duration": float(len(part) + 1),
            "samples": n_samples,
            "chunks": int(np.ceil(n_samples / (CHUNK_S * UI_RATE))),
            "seizures": crises[["start_s", "end_s"]].round(2).values.tolist(),
            "trace_start": crises["trace_start"].iloc[0],
            "windows": len(part),
            "crisis_windows": int(part["Class"].sum()),
            "offset": int(idx[0]),
        })

    windows = {
        "recording": [recording_ids.index(r) for r in dataset["recording_id"]],
        "start": dataset["start_s"].astype(int).tolist(),
        "cls": dataset["Class"].astype(int).tolist(),
        "frac": sig(dataset["crisis_fraction"], 3),
        "test": in_test.tolist(),
        "fold": fold_of.tolist(),
        "features": {f: sig(dataset[f], 4) for f in FEATURES},
        "proba": {name: [round(float(p), 3) for p in oof[name]] for name in names},
        "leaf": leaves,
    }

    meta = {
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "signal": {"rate": UI_RATE, "step_uv": AMP_STEP, "chunk_s": CHUNK_S,
                   "channels": [c.replace("EEG", "") for c in EEG_CHANNELS]},
        "bands": BANDS,
        "features": FEATURES,
        "models": names,
        "recordings": recordings,
        "excluded": ["200625A_F_0000", "210204B_C_0000", "210208B_G_0000"],
        "issues": annotation_issues(ROOT).to_dict(orient="records"),
        "seizure_stats": stats.reset_index().to_dict(orient="records"),
        "windows": windows,
        "results": {
            "split": {"train": int(len(y_train)), "test": int(len(y_test)),
                      "train_pos": int(y_train.sum()), "test_pos": int(y_test.sum())},
            "test": {name: {k: round(float(v), 4) for k, v in test_table.loc[name].items()}
                     for name in test_table.index},
            "confusion": {name: {k: int(v) for k, v in confusions.loc[name].items()}
                          for name in confusions.index},
            "cv": {name: {"f1": round(t.cv_f1, 4), "std": round(t.cv_f1_std, 4),
                          "params": {k: (v if isinstance(v, (int, float, str)) or v is None else str(v))
                                     for k, v in t.best_params.items()},
                          "fit_s": round(t.fit_time_s, 3), "candidates": t.n_candidates}
                   for name, t in tuned.items()},
            "group": {name: {k: round(float(v), 4) for k, v in m.items()} for name, m in group_metrics.items()},
            "depth_curve": curve.assign(max_depth=curve["max_depth"].astype(str)).round(4).to_dict(orient="records"),
        },
        "trees": {
            "gini": tree_json(tuned["Decision Tree (Gini)"].estimator),
            "entropy": tree_json(tuned["Decision Tree (Entropie)"].estimator),
        },
        "importance": {
            "Decision Tree (Gini)": sig(tuned["Decision Tree (Gini)"].estimator.feature_importances_, 3),
            "Decision Tree (Entropie)": sig(tuned["Decision Tree (Entropie)"].estimator.feature_importances_, 3),
            "Random Forest": sig(rf.feature_importances_, 3),
        },
        "comparison": {
            f: {"median0": float(comparison.loc[f, "mediane classe 0"]),
                "median1": float(comparison.loc[f, "mediane classe 1"]),
                "mean0": float(comparison.loc[f, "moyenne classe 0"]),
                "mean1": float(comparison.loc[f, "moyenne classe 1"]),
                "auc": round(float(comparison.loc[f, "AUC (pouvoir discriminant)"]), 4)}
            for f in FEATURES
        },
        "corr": {"pearson": np.round(corr.to_numpy(), 3).tolist(),
                 "spearman": np.round(dataset[FEATURES].corr("spearman").to_numpy(), 3).tolist()},
    }

    meta = no_nan(meta)
    (OUT / "meta.json").write_text(json.dumps(meta, allow_nan=False, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"meta.json : {(OUT / 'meta.json').stat().st_size / 1e6:.1f} Mo")
    for name in names:
        print(f"  {name:26s} F1 test {test_table.loc[name, 'F1-score']:.3f}  F1 par patient {group_metrics[name]['F1-score']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-signal", action="store_true", help="ne pas re-exporter les signaux")
    main(with_signal=not parser.parse_args().no_signal)
