from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import yaml

DEFAULT_DEPLOY_YAML = Path("deploy/robots/g1_getup/config/policy/topology_getup/v0/params/deploy.yaml")


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description=(
      "Update the topology-getup deploy.yaml depth camera contract without touching existing tasks."
    ),
  )
  parser.add_argument(
    "--deploy-yaml",
    type=Path,
    default=DEFAULT_DEPLOY_YAML,
    help="Path to the topology-getup deploy.yaml file to modify.",
  )
  parser.add_argument(
    "--topic-name",
    required=True,
    help="PointCloud2 topic name consumed by the G1 SupportGeometryProvider.",
  )
  parser.add_argument(
    "--pointcloud-mode",
    choices=("euclidean_norm", "z_depth"),
    default=None,
    help="Optional pointcloud projection mode override.",
  )
  parser.add_argument(
    "--cutoff-distance",
    type=float,
    default=None,
    help="Optional cutoff distance override in meters.",
  )
  parser.add_argument(
    "--timeout-ms",
    type=int,
    default=None,
    help="Optional PointCloud2 subscription timeout override in milliseconds.",
  )
  parser.add_argument(
    "--retain-last-valid-frame",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Whether the deploy runtime should retain the last valid SGI frame across depth dropouts.",
  )
  parser.add_argument(
    "--x-field-name",
    default=None,
    help="Optional PointCloud2 x-field name override.",
  )
  parser.add_argument(
    "--y-field-name",
    default=None,
    help="Optional PointCloud2 y-field name override.",
  )
  parser.add_argument(
    "--z-field-name",
    default=None,
    help="Optional PointCloud2 z-field name override.",
  )
  return parser


def update_depth_topic(
  *,
  deploy_yaml: Path,
  topic_name: str,
  pointcloud_mode: str | None = None,
  cutoff_distance: float | None = None,
  timeout_ms: int | None = None,
  retain_last_valid_frame: bool | None = None,
  x_field_name: str | None = None,
  y_field_name: str | None = None,
  z_field_name: str | None = None,
) -> Path:
  payload = yaml.safe_load(deploy_yaml.read_text())
  depth_camera = payload["support_geometry_interface"]["depth_camera"]
  depth_camera["topic_name"] = topic_name
  if pointcloud_mode is not None:
    depth_camera["pointcloud_mode"] = pointcloud_mode
  if cutoff_distance is not None:
    depth_camera["cutoff_distance"] = cutoff_distance
  if timeout_ms is not None:
    depth_camera["timeout_ms"] = timeout_ms
  if retain_last_valid_frame is not None:
    depth_camera["retain_last_valid_frame"] = retain_last_valid_frame
  if any(field_name is not None for field_name in (x_field_name, y_field_name, z_field_name)):
    field_names = depth_camera.setdefault("pointcloud_field_names", {})
    if x_field_name is not None:
      field_names["x"] = x_field_name
    if y_field_name is not None:
      field_names["y"] = y_field_name
    if z_field_name is not None:
      field_names["z"] = z_field_name
  deploy_yaml.write_text(yaml.safe_dump(payload, sort_keys=False))
  return deploy_yaml


def main(argv: Sequence[str] | None = None) -> int:
  parser = build_parser()
  args = parser.parse_args(argv)
  update_depth_topic(
    deploy_yaml=args.deploy_yaml,
    topic_name=args.topic_name,
    pointcloud_mode=args.pointcloud_mode,
    cutoff_distance=args.cutoff_distance,
    timeout_ms=args.timeout_ms,
    retain_last_valid_frame=args.retain_last_valid_frame,
    x_field_name=args.x_field_name,
    y_field_name=args.y_field_name,
    z_field_name=args.z_field_name,
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
