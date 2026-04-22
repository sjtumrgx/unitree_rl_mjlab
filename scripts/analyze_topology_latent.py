from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def _decode_labels(raw: np.ndarray) -> np.ndarray:
  if raw.dtype.kind in {"U", "S"}:
    return raw.astype(str)
  if raw.dtype == object:
    return np.asarray([str(x) for x in raw.tolist()], dtype=str)
  return raw.astype(str)


def load_latent_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
  payload = np.load(path, allow_pickle=True)
  if "topology_latent" not in payload:
    raise ValueError("Expected 'topology_latent' array in NPZ payload.")
  if "bucket" not in payload:
    raise ValueError("Expected 'bucket' labels in NPZ payload.")
  latents = np.asarray(payload["topology_latent"], dtype=np.float32)
  if latents.ndim != 2:
    raise ValueError("'topology_latent' must have shape [N, D].")
  labels = _decode_labels(np.asarray(payload["bucket"]))
  if labels.shape[0] != latents.shape[0]:
    raise ValueError("'bucket' length must match latent batch size.")
  return latents, labels


def summarize_latents(latents: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
  unique_labels = sorted(set(labels.tolist()))
  centroids: dict[str, np.ndarray] = {}
  bucket_counts: dict[str, int] = {}
  within_scatter: dict[str, float] = {}

  for label in unique_labels:
    mask = labels == label
    bucket_latents = latents[mask]
    centroid = bucket_latents.mean(axis=0)
    centroids[label] = centroid
    bucket_counts[label] = int(mask.sum())
    within_scatter[label] = float(np.linalg.norm(bucket_latents - centroid, axis=1).mean())

  centroid_distance: dict[str, dict[str, float]] = {label: {} for label in unique_labels}
  for label_a in unique_labels:
    for label_b in unique_labels:
      if label_a == label_b:
        continue
      centroid_distance[label_a][label_b] = float(
        np.linalg.norm(centroids[label_a] - centroids[label_b])
      )

  correct = 0
  for latent, label in zip(latents, labels, strict=False):
    predicted = min(
      unique_labels,
      key=lambda candidate: float(np.linalg.norm(latent - centroids[candidate])),
    )
    if predicted == label:
      correct += 1

  return {
    "num_samples": int(latents.shape[0]),
    "latent_dim": int(latents.shape[1]),
    "nearest_centroid_accuracy": float(correct / max(len(labels), 1)),
    "bucket_counts": bucket_counts,
    "within_scatter": within_scatter,
    "centroid_distance": centroid_distance,
  }


def _print_json(payload: Any, *, output: Path | None = None) -> None:
  text = json.dumps(payload, indent=2, sort_keys=True)
  if output is not None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
  print(text)


def save_mechanism_plots(summary: dict[str, Any], output_dir: Path) -> None:
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  output_dir.mkdir(parents=True, exist_ok=True)

  labels = list(summary["bucket_counts"])
  heatmap = np.zeros((len(labels), len(labels)), dtype=np.float32)
  for row, label_a in enumerate(labels):
    for col, label_b in enumerate(labels):
      if label_a == label_b:
        continue
      heatmap[row, col] = float(summary["centroid_distance"][label_a][label_b])

  fig, ax = plt.subplots(figsize=(4, 4))
  im = ax.imshow(heatmap, cmap="viridis")
  ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
  ax.set_yticks(range(len(labels)), labels=labels)
  ax.set_title("Centroid distance by topology family")
  fig.colorbar(im, ax=ax)
  fig.tight_layout()
  fig.savefig(output_dir / "centroid_distance_heatmap.png", dpi=160)
  plt.close(fig)

  scatter_items = summary["within_scatter"]
  fig, ax = plt.subplots(figsize=(5, 3))
  ax.bar(list(scatter_items), [float(scatter_items[label]) for label in scatter_items])
  ax.set_title("Within-family latent scatter")
  ax.set_ylabel("Mean distance to centroid")
  ax.tick_params(axis="x", rotation=45)
  fig.tight_layout()
  fig.savefig(output_dir / "within_scatter.png", dpi=160)
  plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Summarize topology-latent NPZ exports for mechanism-analysis workflows.",
  )
  parser.add_argument("latent_npz", type=Path)
  parser.add_argument("--output", type=Path)
  parser.add_argument("--plot-dir", type=Path)
  return parser


def main(argv: Sequence[str] | None = None) -> int:
  parser = build_parser()
  args = parser.parse_args(argv)
  latents, labels = load_latent_dataset(args.latent_npz)
  summary = summarize_latents(latents, labels)
  if args.plot_dir is not None:
    save_mechanism_plots(summary, args.plot_dir)
  _print_json(summary, output=args.output)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
