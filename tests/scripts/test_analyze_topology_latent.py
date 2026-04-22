from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "analyze_topology_latent.py"


def _load_module(module_name: str | None = None):
  unique_name = module_name or f"test_analyze_topology_latent_{len(sys.modules)}"
  sys.modules.pop(unique_name, None)
  spec = importlib.util.spec_from_file_location(unique_name, SCRIPT_PATH)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def _invoke_main(module, args: list[str]):
  main = getattr(module, "main")
  parameters = inspect.signature(main).parameters
  if len(parameters) == 0:
    old_argv = sys.argv
    sys.argv = [str(SCRIPT_PATH), *args]
    try:
      return main()
    finally:
      sys.argv = old_argv
  return main(args)


def test_summarize_latents_reports_bucket_statistics() -> None:
  module = _load_module()
  latents = np.array(
    [
      [0.0, 0.0],
      [0.1, 0.0],
      [1.0, 1.0],
      [1.1, 1.0],
    ],
    dtype=np.float32,
  )
  labels = np.array(["flat", "flat", "stairs", "stairs"], dtype=object)
  summary = module.summarize_latents(latents, labels)
  assert summary["num_samples"] == 4
  assert summary["latent_dim"] == 2
  assert summary["nearest_centroid_accuracy"] == 1.0
  assert summary["bucket_counts"] == {"flat": 2, "stairs": 2}
  assert summary["centroid_distance"]["flat"]["stairs"] > 1.0


def test_main_writes_json_summary(tmp_path: Path, capsys) -> None:
  module = _load_module()
  npz_path = tmp_path / "latents.npz"
  output_path = tmp_path / "summary.json"
  np.savez(
    npz_path,
    topology_latent=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
    bucket=np.array(["flat", "stairs"], dtype=object),
  )
  result = _invoke_main(module, [str(npz_path), "--output", str(output_path)])
  assert result == 0
  payload = json.loads(output_path.read_text())
  assert payload["num_samples"] == 2
  assert "nearest_centroid_accuracy" in payload
  stdout_payload = json.loads(capsys.readouterr().out)
  assert stdout_payload["latent_dim"] == 2


def test_main_can_save_mechanism_plots(tmp_path: Path) -> None:
  module = _load_module()
  npz_path = tmp_path / "latents.npz"
  plot_dir = tmp_path / "plots"
  np.savez(
    npz_path,
    topology_latent=np.array([[0.0, 0.0], [0.1, 0.0], [1.0, 1.0], [1.1, 1.0]], dtype=np.float32),
    bucket=np.array(["flat", "flat", "stairs", "stairs"], dtype=object),
  )
  result = _invoke_main(module, [str(npz_path), "--plot-dir", str(plot_dir)])
  assert result == 0
  assert (plot_dir / "centroid_distance_heatmap.png").exists()
  assert (plot_dir / "within_scatter.png").exists()
