"""RL configuration for Unitree G1 topology get-up tasks."""

from dataclasses import dataclass, field

from mjlab.rl import RslRlBaseRunnerCfg, RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def unitree_g1_topology_getup_ppo_runner_cfg(stage_name: str = "") -> RslRlOnPolicyRunnerCfg:
  cnn_cfg = {
    "output_channels": [16, 32],
    "kernel_size": [5, 3],
    "stride": [2, 2],
    "padding": "zeros",
    "activation": "elu",
    "max_pool": False,
    "global_pool": "none",
    "spatial_softmax": True,
    "spatial_softmax_temperature": 1.0,
    "bottleneck_dim": 64,
  }
  class_name = "src.tasks.velocity.rl.topology_bottleneck_model:TopologyBottleneckCNNModel"
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      cnn_cfg=cnn_cfg,
      class_name=class_name,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      cnn_cfg=cnn_cfg,
      class_name=class_name,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_topology_getup",
    run_name=stage_name,
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10001,
    obs_groups={
      "actor": ("actor", "camera"),
      "critic": ("critic", "camera"),
    },
  )


def unitree_g1_topology_getup_naive_ppo_runner_cfg(stage_name: str = "naive_depth") -> RslRlOnPolicyRunnerCfg:
  cnn_cfg = {
    "output_channels": [16, 32],
    "kernel_size": [5, 3],
    "stride": [2, 2],
    "padding": "zeros",
    "activation": "elu",
    "max_pool": False,
    "global_pool": "none",
    "spatial_softmax": True,
    "spatial_softmax_temperature": 1.0,
  }
  class_name = "mjlab.rl.spatial_softmax:SpatialSoftmaxCNNModel"
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      cnn_cfg=cnn_cfg,
      class_name=class_name,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      cnn_cfg=cnn_cfg,
      class_name=class_name,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_topology_getup_naive",
    run_name=stage_name,
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10001,
    obs_groups={
      "actor": ("actor", "camera"),
      "critic": ("critic", "camera"),
    },
  )


def unitree_g1_topology_getup_teacher_ppo_runner_cfg(stage_name: str = "teacher") -> RslRlOnPolicyRunnerCfg:
  cnn_cfg = {
    "output_channels": [32, 64],
    "kernel_size": [5, 3],
    "stride": [2, 2],
    "padding": "zeros",
    "activation": "elu",
    "max_pool": False,
    "global_pool": "none",
    "spatial_softmax": True,
    "spatial_softmax_temperature": 1.0,
  }
  class_name = "mjlab.rl.spatial_softmax:SpatialSoftmaxCNNModel"
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      cnn_cfg=cnn_cfg,
      class_name=class_name,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      cnn_cfg=cnn_cfg,
      class_name=class_name,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_topology_getup_teacher",
    run_name=stage_name,
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10001,
    obs_groups={
      "actor": ("critic", "camera"),
      "critic": ("critic", "camera"),
    },
  )


@dataclass
class RslRlDistillationAlgorithmCfg:
  class_name: str = "Distillation"
  num_learning_epochs: int = 1
  gradient_length: int = 15
  learning_rate: float = 1.0e-3
  max_grad_norm: float | None = 1.0
  loss_type: str = "mse"
  optimizer: str = "adam"


@dataclass
class TopologyGetupDistillationRunnerCfg(RslRlBaseRunnerCfg):
  class_name: str = "DistillationRunner"
  obs_groups: dict[str, tuple[str, ...]] = field(
    default_factory=lambda: {"student": ("actor", "camera"), "teacher": ("critic", "camera")}
  )
  student: RslRlModelCfg = field(default_factory=lambda: RslRlModelCfg(
    hidden_dims=(512, 256, 128),
    activation="elu",
    obs_normalization=True,
    cnn_cfg={
      "output_channels": [16, 32],
      "kernel_size": [5, 3],
      "stride": [2, 2],
      "padding": "zeros",
      "activation": "elu",
      "max_pool": False,
      "global_pool": "none",
      "spatial_softmax": True,
      "spatial_softmax_temperature": 1.0,
      "bottleneck_dim": 64,
    },
    class_name="src.tasks.velocity.rl.topology_bottleneck_model:TopologyBottleneckCNNModel",
    distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
  ))
  teacher: RslRlModelCfg = field(default_factory=lambda: RslRlModelCfg(
    hidden_dims=(512, 256, 128),
    activation="elu",
    obs_normalization=True,
    cnn_cfg={
      "output_channels": [32, 64],
      "kernel_size": [5, 3],
      "stride": [2, 2],
      "padding": "zeros",
      "activation": "elu",
      "max_pool": False,
      "global_pool": "none",
      "spatial_softmax": True,
      "spatial_softmax_temperature": 1.0,
    },
    class_name="mjlab.rl.spatial_softmax:SpatialSoftmaxCNNModel",
  ))
  algorithm: RslRlDistillationAlgorithmCfg = field(default_factory=RslRlDistillationAlgorithmCfg)
  teacher_load_path: str = ""


def unitree_g1_topology_getup_distillation_runner_cfg(stage_name: str = "distill") -> TopologyGetupDistillationRunnerCfg:
  cfg = TopologyGetupDistillationRunnerCfg()
  cfg.seed = 42
  cfg.num_steps_per_env = 24
  cfg.max_iterations = 5000
  cfg.save_interval = 100
  cfg.experiment_name = "g1_topology_getup_distill"
  cfg.run_name = stage_name
  return cfg
