"""ONNX policy adapter for the exported G1 parkour depth-conditioned policy."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.parkour.contract import (
  ACTOR_INPUT_SHAPE,
  ACTOR_OUTPUT_SHAPE,
  DEPTH_ENCODER_INPUT_SHAPE,
  DEPTH_ENCODER_OUTPUT_SHAPE,
  PROPRIO_SIZE,
  ParkourModelPaths,
  resolve_model_paths,
  shape_as_tuple,
  validate_model_files,
  vector_stats,
)


class ParkourOnnxError(RuntimeError):
  """Raised when the parkour ONNX bundle violates the expected contract."""


@dataclass(frozen=True)
class ParkourOnnxMetadata:
  depth_input_name: str
  depth_output_name: str
  actor_input_name: str
  actor_output_name: str
  depth_input_shape: tuple[int, ...]
  depth_output_shape: tuple[int, ...]
  actor_input_shape: tuple[int, ...]
  actor_output_shape: tuple[int, ...]

  def as_dict(self) -> dict[str, Any]:
    return {
      "depth_input_name": self.depth_input_name,
      "depth_output_name": self.depth_output_name,
      "actor_input_name": self.actor_input_name,
      "actor_output_name": self.actor_output_name,
      "depth_input_shape": list(self.depth_input_shape),
      "depth_output_shape": list(self.depth_output_shape),
      "actor_input_shape": list(self.actor_input_shape),
      "actor_output_shape": list(self.actor_output_shape),
    }


@dataclass(frozen=True)
class ParkourPolicyOutput:
  action: np.ndarray
  depth_latent: np.ndarray
  actor_input: np.ndarray
  diagnostics: Mapping[str, Any]


def _import_onnxruntime():
  try:
    import onnxruntime as ort  # type: ignore[import-not-found]
  except ModuleNotFoundError as exc:  # pragma: no cover - depends on runtime env
    env_name = os.environ.get("G1_PARKOUR_ONNXRUNTIME_CONDA_ENV", "instinct51")
    try:
      prefix = subprocess.check_output(
        [
          "conda",
          "run",
          "-n",
          env_name,
          "python",
          "-c",
          "import sys; print(sys.prefix)",
        ],
        text=True,
        stderr=subprocess.DEVNULL,
      ).strip()
      site_packages = Path(prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
      if site_packages.exists():
        sys.path.insert(0, str(site_packages))
        import onnxruntime as ort  # type: ignore[import-not-found,no-redef]

        return ort
    except Exception:
      pass
    raise ParkourOnnxError(
      "onnxruntime is required for G1 parkour ONNX play. Install it in the "
      "current MJLab environment or expose an existing conda env with "
      "`G1_PARKOUR_ONNXRUNTIME_CONDA_ENV` (default: instinct51)."
    ) from exc
  return ort


def _session_shape(node: Any) -> tuple[int, ...]:
  try:
    return shape_as_tuple(node.shape)
  except ValueError as exc:
    raise ParkourOnnxError(str(exc)) from exc


def _check_shape(name: str, actual: tuple[int, ...], expected: tuple[int, ...]) -> None:
  if actual != expected:
    raise ParkourOnnxError(
      f"{name} shape mismatch: expected {list(expected)}, got {list(actual)}"
    )


class ParkourOnnxPolicy:
  """Runs ``0-depth_encoder.onnx`` then ``actor.onnx`` like ParkourOrtRunner."""

  def __init__(
    self,
    *,
    policy_dir: Path | str | None = None,
    exported_dir: Path | str | None = None,
    providers: list[str] | None = None,
  ) -> None:
    self.paths: ParkourModelPaths = resolve_model_paths(
      policy_dir=policy_dir,
      exported_dir=exported_dir,
    )
    validate_model_files(self.paths)
    ort = _import_onnxruntime()
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    resolved_providers = providers or ["CPUExecutionProvider"]
    self._depth_session = ort.InferenceSession(
      str(self.paths.depth_encoder_onnx),
      sess_options=session_options,
      providers=resolved_providers,
    )
    self._actor_session = ort.InferenceSession(
      str(self.paths.actor_onnx),
      sess_options=session_options,
      providers=resolved_providers,
    )
    self.metadata = self._read_metadata()
    self.validate_metadata()

  def _read_metadata(self) -> ParkourOnnxMetadata:
    depth_input = self._depth_session.get_inputs()[0]
    depth_output = self._depth_session.get_outputs()[0]
    actor_input = self._actor_session.get_inputs()[0]
    actor_output = self._actor_session.get_outputs()[0]
    return ParkourOnnxMetadata(
      depth_input_name=depth_input.name,
      depth_output_name=depth_output.name,
      actor_input_name=actor_input.name,
      actor_output_name=actor_output.name,
      depth_input_shape=_session_shape(depth_input),
      depth_output_shape=_session_shape(depth_output),
      actor_input_shape=_session_shape(actor_input),
      actor_output_shape=_session_shape(actor_output),
    )

  def validate_metadata(self) -> None:
    _check_shape("depth encoder input", self.metadata.depth_input_shape, DEPTH_ENCODER_INPUT_SHAPE)
    _check_shape("depth encoder output", self.metadata.depth_output_shape, DEPTH_ENCODER_OUTPUT_SHAPE)
    _check_shape("actor input", self.metadata.actor_input_shape, ACTOR_INPUT_SHAPE)
    _check_shape("actor output", self.metadata.actor_output_shape, ACTOR_OUTPUT_SHAPE)

  def act(self, proprio: np.ndarray, depth: np.ndarray) -> ParkourPolicyOutput:
    proprio_arr = np.asarray(proprio, dtype=np.float32).reshape(1, -1)
    depth_arr = np.asarray(depth, dtype=np.float32).reshape(DEPTH_ENCODER_INPUT_SHAPE)
    if proprio_arr.shape != (1, PROPRIO_SIZE):
      raise ParkourOnnxError(
        f"proprio shape mismatch: expected [1, {PROPRIO_SIZE}], got {list(proprio_arr.shape)}"
      )

    depth_outputs = self._depth_session.run(
      [self.metadata.depth_output_name],
      {self.metadata.depth_input_name: depth_arr},
    )
    depth_latent = np.asarray(depth_outputs[0], dtype=np.float32).reshape(DEPTH_ENCODER_OUTPUT_SHAPE)
    actor_input = np.concatenate([proprio_arr, depth_latent], axis=1).astype(np.float32, copy=False)
    if actor_input.shape != ACTOR_INPUT_SHAPE:
      raise ParkourOnnxError(
        f"actor input shape mismatch after concat: expected {ACTOR_INPUT_SHAPE}, got {actor_input.shape}"
      )
    actor_outputs = self._actor_session.run(
      [self.metadata.actor_output_name],
      {self.metadata.actor_input_name: actor_input},
    )
    action = np.asarray(actor_outputs[0], dtype=np.float32).reshape(ACTOR_OUTPUT_SHAPE)
    diagnostics = {
      "proprio_stats": vector_stats(proprio_arr),
      "depth_stats": vector_stats(depth_arr),
      "latent_stats": vector_stats(depth_latent),
      "actor_input_stats": vector_stats(actor_input),
      "action_stats": vector_stats(action),
    }
    return ParkourPolicyOutput(
      action=action.reshape(-1),
      depth_latent=depth_latent.reshape(-1),
      actor_input=actor_input.reshape(-1),
      diagnostics=diagnostics,
    )
