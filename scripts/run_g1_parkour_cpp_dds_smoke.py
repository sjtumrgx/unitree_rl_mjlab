#!/usr/bin/env python3
"""Two-process loopback smoke harness for G1 Parkour C++/DDS depth walking.

The harness starts the MuJoCo parkour simulator and the Unitree SDK controller,
captures both logs, and reports machine-readable acceptance markers.  It is a
local runtime harness, not a CI-only unit test: it expects the graphical/runtime
stack needed by ``unitree_mujoco_parkour`` to be available.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue, Empty
from typing import Iterable

REQUIRED_MARKERS = (
  "PUBLISHER_READY",
  "FIRST_VALID_DEPTH_STACK",
  "ENTERED_PARKOUR",
  "NONZERO_COMMAND",
  "ORDER_PARITY_OK",
  "DISTANCE_X",
  "NO_FALL_RESET",
)


@dataclass
class ProcessSpec:
  name: str
  cmd: list[str]
  log_path: Path
  cwd: Path | None = None


@dataclass
class HarnessState:
  walk_distance: float
  required_markers: tuple[str, ...] = REQUIRED_MARKERS
  markers: dict[str, bool] = field(default_factory=lambda: {marker: False for marker in REQUIRED_MARKERS})
  lines: list[tuple[str, str]] = field(default_factory=list)
  distance_x: float = 0.0
  latest_progress: dict[str, float | str | bool] = field(default_factory=dict)
  fall_reset_detected: bool = False
  processes_exited: dict[str, int | None] = field(default_factory=dict)
  sim_ready: bool = False

  def observe(self, source: str, line: str) -> None:
    self.lines.append((source, line.rstrip()))
    if "HEADLESS_READY" in line:
      self.sim_ready = True
    if "PUBLISHER_READY" in line:
      self.markers["PUBLISHER_READY"] = True
    if "FIRST_VALID_DEPTH_STACK" in line:
      self.markers["FIRST_VALID_DEPTH_STACK"] = True
    if "ENTERED_PARKOUR" in line:
      self.markers["ENTERED_PARKOUR"] = True
    if "NONZERO_COMMAND" in line:
      self.markers["NONZERO_COMMAND"] = True
    if "ORDER_PARITY_OK" in line:
      self.markers["ORDER_PARITY_OK"] = True
    if "FALL_RESET_DETECTED" in line or "nan" in line.lower() or "non-finite" in line.lower():
      self.fall_reset_detected = True
    distance_match = re.search(r"DISTANCE_X>=([0-9.+\-eE]+)\s+distance_x=([0-9.+\-eE]+)", line)
    if distance_match:
      self.distance_x = max(self.distance_x, float(distance_match.group(2)))
      if self.distance_x >= self.walk_distance:
        self.markers["DISTANCE_X"] = True
    else:
      numeric_match = re.search(r"distance_x=([0-9.+\-eE]+)", line)
      if numeric_match:
        self.distance_x = max(self.distance_x, float(numeric_match.group(1)))
        if self.distance_x >= self.walk_distance:
          self.markers["DISTANCE_X"] = True
    if "PARKOUR_PROGRESS" in line:
      self.latest_progress = _parse_key_value_line(line)
    if "NO_FALL_RESET" in line or not self.fall_reset_detected:
      self.markers["NO_FALL_RESET"] = True

  def success(self) -> bool:
    self.markers["NO_FALL_RESET"] = not self.fall_reset_detected
    return all(self.markers[marker] for marker in self.required_markers)


def _reader_thread(proc: subprocess.Popen[str], spec: ProcessSpec, queue: Queue[tuple[str, str]], log_file) -> None:
  assert proc.stdout is not None
  for line in proc.stdout:
    log_file.write(line)
    log_file.flush()
    queue.put((spec.name, line))


def _start_process(
  spec: ProcessSpec,
  env: dict[str, str],
  output_queue: Queue[tuple[str, str]],
) -> tuple[subprocess.Popen[str], object, threading.Thread]:
  spec.log_path.parent.mkdir(parents=True, exist_ok=True)
  log_file = spec.log_path.open("w", encoding="utf-8")
  proc = subprocess.Popen(
    spec.cmd,
    cwd=str(spec.cwd) if spec.cwd else None,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
    env=env,
    preexec_fn=os.setsid if os.name != "nt" else None,
  )
  return proc, log_file, threading.Thread(target=_reader_thread, args=(proc, spec, output_queue, log_file), daemon=True)


def _terminate_process(proc: subprocess.Popen[str]) -> None:
  if proc.poll() is not None:
    return
  try:
    if os.name != "nt":
      os.killpg(proc.pid, signal.SIGTERM)
    else:
      proc.terminate()
    proc.wait(timeout=5)
  except Exception:
    try:
      if os.name != "nt":
        os.killpg(proc.pid, signal.SIGKILL)
      else:
        proc.kill()
    except Exception:
      pass


def _parse_key_value_line(line: str) -> dict[str, float | str | bool]:
  parsed: dict[str, float | str | bool] = {}
  for token in line.strip().split():
    if "=" not in token:
      continue
    key, value = token.split("=", 1)
    if value.lower() in {"true", "false"}:
      parsed[key] = value.lower() == "true"
      continue
    try:
      parsed[key] = float(value)
    except ValueError:
      parsed[key] = value
  return parsed


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--sim-bin", type=Path, default=Path("simulate/build/unitree_mujoco_parkour"))
  parser.add_argument("--ctrl-bin", type=Path, default=Path("deploy/robots/g1_parkour/build/g1_parkour_ctrl"))
  parser.add_argument("--network", default="lo")
  parser.add_argument("--sim-autostart-parkour", action="store_true", default=True)
  parser.add_argument("--no-sim-autostart-parkour", dest="sim_autostart_parkour", action="store_false")
  parser.add_argument("--sim-command-x", type=float, default=0.25)
  parser.add_argument("--sim-command-y", type=float, default=0.0)
  parser.add_argument("--sim-command-yaw", type=float, default=0.0)
  parser.add_argument("--no-sim-heading-lock", action="store_true", help="Pass --no-sim-heading-lock to the controller.")
  parser.add_argument("--walk-distance", type=float, default=5.0)
  parser.add_argument("--timeout-seconds", type=float, default=90.0)
  parser.add_argument("--progress-log-interval", type=float, default=1.0)
  parser.add_argument("--controller-start-delay", type=float, default=2.0)
  parser.add_argument("--log-dir", type=Path, default=None)
  parser.add_argument("--hide-depth-debug-window", action="store_true", help="Use hidden render context for depth bridge debug window.")
  parser.add_argument(
    "--disable-depth-bridge",
    action="store_true",
    help="Set G1_PARKOUR_DEPTH_BRIDGE=0 to disable simulator-side live depth rendering/publishing.",
  )
  parser.add_argument("--headless-sim", action="store_true", help="Run simulator with --headless for control-only diagnostics.")
  parser.add_argument("--sim-scene", type=Path, help="Optional MuJoCo scene XML/MJB to pass through to the simulator.")
  parser.add_argument("--constant-depth", type=float, default=None, help="Set G1_PARKOUR_DEBUG_CONSTANT_DEPTH for controller diagnostics.")
  parser.add_argument(
    "--ctrl-gait-record-jsonl",
    type=Path,
    help="Pass --gait-record-jsonl to the controller for C++/DDS gait parity capture.",
  )
  parser.add_argument(
    "--ctrl-gait-record-every",
    type=int,
    default=1,
    help="Pass --gait-record-every to the controller when gait recording is enabled.",
  )
  return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
  args = parse_args(argv)
  root = Path(__file__).resolve().parents[1]
  log_dir = args.log_dir or args.ctrl_bin.resolve().parent / "cpp_dds_smoke_logs"
  log_dir.mkdir(parents=True, exist_ok=True)

  sim_bin = args.sim_bin if args.sim_bin.is_absolute() else root / args.sim_bin
  ctrl_bin = args.ctrl_bin if args.ctrl_bin.is_absolute() else root / args.ctrl_bin
  if not sim_bin.exists():
    raise FileNotFoundError(f"sim binary not found: {sim_bin}")
  if not ctrl_bin.exists():
    raise FileNotFoundError(f"controller binary not found: {ctrl_bin}")

  env = os.environ.copy()
  if args.hide_depth_debug_window:
    env["G1_PARKOUR_DEPTH_DEBUG_WINDOW"] = "0"
  if args.disable_depth_bridge:
    env["G1_PARKOUR_DEPTH_BRIDGE"] = "0"
  if args.constant_depth is not None:
    env["G1_PARKOUR_DEBUG_CONSTANT_DEPTH"] = str(args.constant_depth)

  required_markers = REQUIRED_MARKERS
  if (args.headless_sim or args.disable_depth_bridge) and args.constant_depth is not None:
    required_markers = tuple(marker for marker in REQUIRED_MARKERS if marker != "PUBLISHER_READY")

  sim_cmd = [str(sim_bin), "--network", args.network]
  if args.sim_scene is not None:
    sim_scene = args.sim_scene if args.sim_scene.is_absolute() else root / args.sim_scene
    sim_cmd += ["--scene", str(sim_scene)]
  if args.headless_sim:
    sim_cmd += ["--headless", "--headless-seconds", str(args.timeout_seconds)]
  sim_cmd += [
    "--walk-distance-marker", str(args.walk_distance),
    "--progress-log-interval", str(args.progress_log_interval),
  ]
  sim_spec = ProcessSpec(
    name="sim",
    cmd=sim_cmd,
    log_path=log_dir / "sim.log",
    cwd=sim_bin.parent,
  )
  ctrl_cmd = [str(ctrl_bin), "--network", args.network]
  if args.sim_autostart_parkour:
    ctrl_cmd += [
      "--sim-autostart-parkour",
      "--sim-command-x", str(args.sim_command_x),
      "--sim-command-y", str(args.sim_command_y),
      "--sim-command-yaw", str(args.sim_command_yaw),
    ]
    if args.no_sim_heading_lock:
      ctrl_cmd += ["--no-sim-heading-lock"]
  ctrl_gait_record_path: Path | None = None
  if args.ctrl_gait_record_jsonl is not None:
    ctrl_gait_record_path = args.ctrl_gait_record_jsonl
    if not ctrl_gait_record_path.is_absolute():
      ctrl_gait_record_path = root / ctrl_gait_record_path
    ctrl_cmd += [
      "--gait-record-jsonl", str(ctrl_gait_record_path),
      "--gait-record-every", str(max(1, args.ctrl_gait_record_every)),
    ]
  ctrl_spec = ProcessSpec(
    name="ctrl",
    cmd=ctrl_cmd,
    log_path=log_dir / "ctrl.log",
    cwd=ctrl_bin.parent,
  )

  state = HarnessState(walk_distance=args.walk_distance, required_markers=required_markers)
  output_queue: Queue[tuple[str, str]] = Queue()
  processes: list[subprocess.Popen[str]] = []
  log_files: list[object] = []
  threads: list[threading.Thread] = []
  start = time.monotonic()
  try:
    sim_proc, sim_log, sim_thread = _start_process(sim_spec, env, output_queue)
    processes.append(sim_proc)
    log_files.append(sim_log)
    sim_thread.start()
    threads.append(sim_thread)

    # Let simulator load model/Unitree bridge and start publisher before controller connects.
    ctrl_started = False
    while time.monotonic() - start < args.timeout_seconds:
      try:
        source, line = output_queue.get(timeout=0.1)
        state.observe(source, line)
        print(f"[{source}] {line}", end="")
      except Empty:
        pass
      sim_rc = sim_proc.poll()
      if not ctrl_started and sim_rc is not None:
        state.processes_exited[sim_spec.name] = sim_rc
        break
      if not ctrl_started and (
        state.markers["PUBLISHER_READY"]
        or state.sim_ready
        or time.monotonic() - start >= args.controller_start_delay
      ):
        ctrl_proc, ctrl_log, ctrl_thread = _start_process(ctrl_spec, env, output_queue)
        processes.append(ctrl_proc)
        log_files.append(ctrl_log)
        ctrl_thread.start()
        threads.append(ctrl_thread)
        ctrl_started = True
      for spec, proc in zip((sim_spec, ctrl_spec), processes):
        rc = proc.poll()
        if rc is not None:
          state.processes_exited[spec.name] = rc
      if state.fall_reset_detected:
        break
      if state.success():
        break

    while True:
      try:
        source, line = output_queue.get_nowait()
      except Empty:
        break
      state.observe(source, line)
      print(f"[{source}] {line}", end="")

  finally:
    for proc in processes:
      _terminate_process(proc)
    for log_file in log_files:
      try:
        log_file.close()
      except Exception:
        pass

  state.markers["NO_FALL_RESET"] = not state.fall_reset_detected
  summary = {
    "status": "ok" if state.success() else "failed",
    "markers": state.markers,
    "required_markers": list(state.required_markers),
    "walk_distance_target": args.walk_distance,
    "distance_x": state.distance_x,
    "latest_progress": state.latest_progress,
    "fall_reset_detected": state.fall_reset_detected,
    "processes_exited": state.processes_exited,
    "commands": {"sim": sim_spec.cmd, "ctrl": ctrl_spec.cmd},
    "logs": {"sim": str(sim_spec.log_path), "ctrl": str(ctrl_spec.log_path)},
    "gait_record": str(ctrl_gait_record_path) if ctrl_gait_record_path is not None else None,
  }
  (log_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
  print(json.dumps(summary, indent=2, ensure_ascii=False))
  return 0 if state.success() else 1


if __name__ == "__main__":
  raise SystemExit(main())
