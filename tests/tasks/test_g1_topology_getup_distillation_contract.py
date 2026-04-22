from __future__ import annotations

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls

from src.tasks.velocity.rl.topology_getup_distillation_runner import TopologyGetupDistillationRunner


def test_distillation_task_is_registered_with_dedicated_runner() -> None:
  assert load_runner_cls("Unitree-G1-TopologyGetUp-Stage0-Distill") is TopologyGetupDistillationRunner


def test_distillation_cfg_freezes_student_teacher_split_and_requires_checkpoint_path_field() -> None:
  cfg = load_rl_cfg("Unitree-G1-TopologyGetUp-Stage0-Distill")
  assert cfg.experiment_name == "g1_topology_getup_distill"
  assert cfg.obs_groups == {"student": ("actor", "camera"), "teacher": ("critic", "camera")}
  assert cfg.student.class_name == "src.tasks.velocity.rl.topology_bottleneck_model:TopologyBottleneckCNNModel"
  assert cfg.teacher.class_name == "mjlab.rl.spatial_softmax:SpatialSoftmaxCNNModel"
  assert cfg.algorithm.class_name == "Distillation"
  assert cfg.teacher_load_path == ""
