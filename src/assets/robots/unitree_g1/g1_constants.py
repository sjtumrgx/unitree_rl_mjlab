"""Unitree G1 constants."""

from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import (
  ElectricActuator,
  reflected_inertia_from_two_stage_planetary,
)
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

G1_XML: Path = (
  SRC_PATH / "assets" / "robots" / "unitree_g1" / "xmls" / "g1.xml"
)
assert G1_XML.exists()

G1_PARKOUR_XML: Path = (
  SRC_PATH / "assets" / "robots" / "unitree_g1" / "xmls" / "scene_g1_parkour.xml"
)
assert G1_PARKOUR_XML.exists()


def get_assets(meshdir: str, *, xml_path: Path = G1_XML) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, xml_path.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(G1_XML))
  spec.assets = get_assets(spec.meshdir, xml_path=G1_XML)
  return spec


def get_g1_parkour_flat_debug_spec() -> mujoco.MjSpec:
  """Load the parkour-training G1 XML as a robot-only MJLab entity spec.

  ``scene_g1_parkour.xml`` intentionally contains the torso-root parkour robot,
  depth camera, IMU sensors, XML torque motors, and a small obstacle course for
  the standalone Unitree simulator.  MJLab entity configs add their own
  actuators and terrain when the robot is attached to a scene, so the first-stage
  flat-debug task strips the standalone motors and world obstacles while keeping
  the torso-root robot, camera, collision proxies, and parkour IMU sensors.
  """
  spec = mujoco.MjSpec.from_file(str(G1_PARKOUR_XML))
  spec.assets = get_assets(spec.meshdir, xml_path=G1_PARKOUR_XML)

  for actuator in list(spec.actuators):
    spec.delete(actuator)

  for body in list(spec.worldbody.bodies):
    if body.name != "torso_link":
      spec.delete(body)

  for geom in list(spec.worldbody.geoms):
    spec.delete(geom)

  # The standalone MuJoCo scene carries XML-default passive joint properties
  # used by the C++ lowcmd simulator.  MJLab adds its own actuator armature,
  # damping, and stiffness from ``G1_ARTICULATION`` below; keeping the XML
  # defaults would add root/free-joint damping/friction and duplicate hinge
  # damping relative to the IsaacLab training contract.
  for joint in spec.joints:
    joint.damping[:] = 0.0
    joint.armature = 0.0
    joint.frictionloss = 0.0

  return spec


def _add_parkour_obstacle_debug_geoms(spec: mujoco.MjSpec) -> None:
  """Add a conservative low block + shallow gap visual/contact course.

  The first renderer-depth gate intentionally uses tiny deterministic features:
  a 5 cm block and a 10 cm lower strip between two 5 cm platforms.  The flat
  terrain plane remains below the gap so the scene is easy to traverse while the
  head camera still sees an objective depth discontinuity.
  """
  spec.worldbody.add_geom(
    name="parkour_debug_low_block",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=[1.05, 0.0, 0.025],
    size=[0.10, 0.45, 0.025],
    rgba=[0.70, 0.55, 0.35, 1.0],
  )
  spec.worldbody.add_geom(
    name="parkour_debug_gap_near_lip",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=[1.55, 0.0, 0.025],
    size=[0.10, 0.45, 0.025],
    rgba=[0.45, 0.45, 0.50, 1.0],
  )
  spec.worldbody.add_geom(
    name="parkour_debug_gap_far_lip",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=[1.85, 0.0, 0.025],
    size=[0.10, 0.45, 0.025],
    rgba=[0.45, 0.45, 0.50, 1.0],
  )


PARKOUR_COMPLEX_TERRAIN_GEOM_NAMES = (
  "parkour_complex_up_stair_01",
  "parkour_complex_up_stair_02",
  "parkour_complex_up_stair_03",
  "parkour_complex_up_stair_04",
  "parkour_complex_up_stair_05",
  "parkour_complex_top_platform",
  "parkour_complex_down_stair_01",
  "parkour_complex_down_stair_02",
  "parkour_complex_down_stair_03",
  "parkour_complex_down_stair_04",
  "parkour_complex_down_stair_05",
  "parkour_complex_gap_near_platform",
  "parkour_complex_gap_floor_marker",
  "parkour_complex_gap_far_platform",
  "parkour_complex_discrete_box_01",
  "parkour_complex_discrete_box_02",
  "parkour_complex_discrete_box_03",
  "parkour_complex_discrete_box_04",
  "parkour_complex_discrete_box_05",
  "parkour_complex_discrete_box_06",
  "parkour_complex_up_stair_b_01",
  "parkour_complex_up_stair_b_02",
  "parkour_complex_up_stair_b_03",
  "parkour_complex_up_stair_b_04",
  "parkour_complex_mid_platform_b",
  "parkour_complex_down_stair_b_01",
  "parkour_complex_down_stair_b_02",
  "parkour_complex_down_stair_b_03",
  "parkour_complex_down_stair_b_04",
  "parkour_complex_second_gap_near_platform",
  "parkour_complex_second_gap_floor_marker",
  "parkour_complex_second_gap_far_platform",
  "parkour_complex_mesh_box_01",
  "parkour_complex_mesh_box_02",
  "parkour_complex_mesh_box_03",
  "parkour_complex_mesh_box_04",
  "parkour_complex_mesh_box_05",
  "parkour_complex_mesh_box_06",
)


def _add_parkour_complex_terrain_debug_geoms(spec: mujoco.MjSpec) -> None:
  """Add deterministic MuJoCo approximations of InstinctLab parkour terrain.

  InstinctLab's training terrain is procedurally generated from height-fields
  (``pyramid_stairs``, ``pyramid_stairs_inv``, ``square_gaps``, ``boxes`` and
  ``mesh_boxes``).  MJLab's debug path keeps this as explicit MJCF box geoms so
  depth-camera alignment and locomotion regressions are reproducible without
  importing IsaacLab terrain generators.
  """

  def add_box(
    name: str,
    *,
    pos: tuple[float, float, float],
    size: tuple[float, float, float],
    rgba: tuple[float, float, float, float],
  ) -> None:
    spec.worldbody.add_geom(
      name=name,
      type=mujoco.mjtGeom.mjGEOM_BOX,
      pos=list(pos),
      size=list(size),
      rgba=list(rgba),
    )

  stair_length = 0.36
  stair_half_width = 0.72
  up_start_x = 1.20
  # Keep the InstinctLab stair pattern but increase the inter-step height to
  # require more visible foot clearance.  This intentionally makes the
  # complete-course asset harder than the earlier conservative 20 cm course.
  for index, height in enumerate((0.06, 0.12, 0.18, 0.24, 0.30), start=1):
    add_box(
      f"parkour_complex_up_stair_{index:02d}",
      pos=(up_start_x + (index - 1) * stair_length, 0.0, height / 2.0),
      size=(stair_length / 2.0, stair_half_width, height / 2.0),
      rgba=(0.58, 0.50, 0.42, 1.0),
    )

  add_box(
    "parkour_complex_top_platform",
    pos=(3.25, 0.0, 0.15),
    size=(0.42, stair_half_width, 0.15),
    rgba=(0.50, 0.52, 0.55, 1.0),
  )

  down_start_x = 4.20
  for index, height in enumerate((0.30, 0.24, 0.18, 0.12, 0.06), start=1):
    add_box(
      f"parkour_complex_down_stair_{index:02d}",
      pos=(down_start_x + (index - 1) * stair_length, 0.0, height / 2.0),
      size=(stair_length / 2.0, stair_half_width, height / 2.0),
      rgba=(0.52, 0.47, 0.40, 1.0),
    )

  # Square-gap approximation: raised lips separated by a lower strip.  The
  # global floor remains intact, so this is safe for early debugging while the
  # 28 cm lip-to-floor drop and 14 cm lower strip keep the surrogate visibly
  # different from stepping across the global floor; debug runs now assert that
  # feet do not contact the lower strip while traversing the gap.
  add_box(
    "parkour_complex_gap_near_platform",
    pos=(6.80, 0.0, 0.14),
    size=(0.35, stair_half_width, 0.14),
    rgba=(0.42, 0.43, 0.48, 1.0),
  )
  add_box(
    "parkour_complex_gap_floor_marker",
    pos=(7.22, 0.0, 0.001),
    size=(0.07, 0.66, 0.001),
    rgba=(0.05, 0.05, 0.06, 1.0),
  )
  add_box(
    "parkour_complex_gap_far_platform",
    pos=(7.64, 0.0, 0.14),
    size=(0.35, stair_half_width, 0.14),
    rgba=(0.42, 0.43, 0.48, 1.0),
  )

  # Discrete boxes roughly mirror InstinctLab's ``boxes`` terrain.  The C++ DDS
  # route currently has no y-waypoint follower, so keep boxes centered and wide
  # enough that the forward command can step over them instead of steering around
  # narrow off-axis blocks.
  for index, (x, y, sx, sy, height) in enumerate(
    (
      (9.30, 0.00, 0.30, 0.72, 0.04),
      (10.00, 0.00, 0.30, 0.72, 0.06),
      (10.80, 0.00, 0.30, 0.72, 0.08),
      (11.60, 0.00, 0.30, 0.72, 0.06),
      (12.35, 0.00, 0.30, 0.72, 0.07),
      (13.15, 0.00, 0.30, 0.72, 0.04),
    ),
    start=1,
  ):
    add_box(
      f"parkour_complex_discrete_box_{index:02d}",
      pos=(x, y, height / 2.0),
      size=(sx, sy, height / 2.0),
      rgba=(0.62, 0.42, 0.28, 1.0),
    )

  second_stair_length = 0.42
  second_up_start_x = 14.40
  for index, height in enumerate((0.06, 0.12, 0.18, 0.24), start=1):
    add_box(
      f"parkour_complex_up_stair_b_{index:02d}",
      pos=(second_up_start_x + (index - 1) * second_stair_length, 0.0, height / 2.0),
      size=(second_stair_length / 2.0, stair_half_width, height / 2.0),
      rgba=(0.56, 0.48, 0.40, 1.0),
    )

  add_box(
    "parkour_complex_mid_platform_b",
    pos=(16.00, 0.0, 0.12),
    size=(0.42, stair_half_width, 0.12),
    rgba=(0.48, 0.50, 0.54, 1.0),
  )

  second_down_start_x = 16.70
  for index, height in enumerate((0.24, 0.18, 0.12, 0.06), start=1):
    add_box(
      f"parkour_complex_down_stair_b_{index:02d}",
      pos=(second_down_start_x + (index - 1) * second_stair_length, 0.0, height / 2.0),
      size=(second_stair_length / 2.0, stair_half_width, height / 2.0),
      rgba=(0.50, 0.44, 0.38, 1.0),
    )

  add_box(
    "parkour_complex_second_gap_near_platform",
    pos=(19.10, 0.0, 0.14),
    size=(0.35, stair_half_width, 0.14),
    rgba=(0.38, 0.40, 0.46, 1.0),
  )
  add_box(
    "parkour_complex_second_gap_floor_marker",
    pos=(19.52, 0.0, 0.001),
    size=(0.07, 0.66, 0.001),
    rgba=(0.04, 0.04, 0.05, 1.0),
  )
  add_box(
    "parkour_complex_second_gap_far_platform",
    pos=(19.94, 0.0, 0.14),
    size=(0.35, stair_half_width, 0.14),
    rgba=(0.38, 0.40, 0.46, 1.0),
  )

  # Mesh-box style stepping stones: smaller blocks close to the center line,
  # inspired by InstinctLab's random multi-box terrain.
  for index, (x, y, sx, sy, height) in enumerate(
    (
      (21.30, 0.00, 0.26, 0.62, 0.04),
      (22.00, 0.00, 0.26, 0.62, 0.05),
      (22.70, 0.00, 0.26, 0.62, 0.06),
      (23.40, 0.00, 0.26, 0.62, 0.05),
      (24.10, 0.00, 0.26, 0.62, 0.06),
      (24.80, 0.00, 0.26, 0.62, 0.04),
    ),
    start=1,
  ):
    add_box(
      f"parkour_complex_mesh_box_{index:02d}",
      pos=(x, y, height / 2.0),
      size=(sx, sy, height / 2.0),
      rgba=(0.34, 0.46, 0.32, 1.0),
    )


def get_g1_parkour_obstacle_debug_spec() -> mujoco.MjSpec:
  """Load the torso-root parkour G1 plus a conservative obstacle course."""
  spec = get_g1_parkour_flat_debug_spec()
  _add_parkour_obstacle_debug_geoms(spec)
  return spec


def get_g1_parkour_complex_terrain_debug_spec() -> mujoco.MjSpec:
  """Load the torso-root parkour G1 plus deterministic complex terrain geoms."""
  spec = get_g1_parkour_flat_debug_spec()
  _add_parkour_complex_terrain_debug_geoms(spec)
  return spec


##
# Actuator config.
##

# Motor specs (from Unitree).
ROTOR_INERTIAS_5020 = (
  0.139e-4,
  0.017e-4,
  0.169e-4,
)
GEARS_5020 = (
  1,
  1 + (46 / 18),
  1 + (56 / 16),
)
ARMATURE_5020 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_5020, GEARS_5020
)

ROTOR_INERTIAS_7520_14 = (
  0.489e-4,
  0.098e-4,
  0.533e-4,
)
GEARS_7520_14 = (
  1,
  4.5,
  1 + (48 / 22),
)
ARMATURE_7520_14 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_7520_14, GEARS_7520_14
)

ROTOR_INERTIAS_7520_22 = (
  0.489e-4,
  0.109e-4,
  0.738e-4,
)
GEARS_7520_22 = (
  1,
  4.5,
  5,
)
ARMATURE_7520_22 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_7520_22, GEARS_7520_22
)

ROTOR_INERTIAS_4010 = (
  0.068e-4,
  0.0,
  0.0,
)
GEARS_4010 = (
  1,
  5,
  5,
)
ARMATURE_4010 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_4010, GEARS_4010
)

ACTUATOR_5020 = ElectricActuator(
  reflected_inertia=ARMATURE_5020,
  velocity_limit=37.0,
  effort_limit=25.0,
)
ACTUATOR_7520_14 = ElectricActuator(
  reflected_inertia=ARMATURE_7520_14,
  velocity_limit=32.0,
  effort_limit=88.0,
)
ACTUATOR_7520_22 = ElectricActuator(
  reflected_inertia=ARMATURE_7520_22,
  velocity_limit=20.0,
  effort_limit=139.0,
)
ACTUATOR_4010 = ElectricActuator(
  reflected_inertia=ARMATURE_4010,
  velocity_limit=22.0,
  effort_limit=5.0,
)

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

G1_ACTUATOR_5020 = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_elbow_joint",
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_wrist_roll_joint",
  ),
  stiffness=STIFFNESS_5020,
  damping=DAMPING_5020,
  effort_limit=ACTUATOR_5020.effort_limit,
  armature=ACTUATOR_5020.reflected_inertia,
)
G1_ACTUATOR_7520_14 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_pitch_joint", ".*_hip_yaw_joint", "waist_yaw_joint"),
  stiffness=STIFFNESS_7520_14,
  damping=DAMPING_7520_14,
  effort_limit=ACTUATOR_7520_14.effort_limit,
  armature=ACTUATOR_7520_14.reflected_inertia,
)
G1_ACTUATOR_7520_22 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_roll_joint", ".*_knee_joint"),
  stiffness=STIFFNESS_7520_22,
  damping=DAMPING_7520_22,
  effort_limit=ACTUATOR_7520_22.effort_limit,
  armature=ACTUATOR_7520_22.reflected_inertia,
)
G1_ACTUATOR_4010 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_wrist_pitch_joint", ".*_wrist_yaw_joint"),
  stiffness=STIFFNESS_4010,
  damping=DAMPING_4010,
  effort_limit=ACTUATOR_4010.effort_limit,
  armature=ACTUATOR_4010.reflected_inertia,
)

# Waist pitch/roll and ankles are 4-bar linkages with 2 5020 actuators.
# Due to the parallel linkage, the effective armature at the ankle and waist joints
# is configuration dependent. Since the exact geometry of the linkage is unknown, we
# assume a nominal 1:1 gear ratio. Under this assumption, the joint armature in the
# nominal configuration is approximated as the sum of the 2 actuators' armatures.
G1_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_pitch_joint", "waist_roll_joint"),
  stiffness=STIFFNESS_5020 * 2,
  damping=DAMPING_5020 * 2,
  effort_limit=ACTUATOR_5020.effort_limit * 2,
  armature=ACTUATOR_5020.reflected_inertia * 2,
)
G1_ACTUATOR_ANKLE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
  stiffness=STIFFNESS_5020 * 2,
  damping=DAMPING_5020 * 2,
  effort_limit=ACTUATOR_5020.effort_limit * 2,
  armature=ACTUATOR_5020.reflected_inertia * 2,
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.8),
  joint_pos={
    ".*_hip_pitch_joint": -0.1,
    ".*_knee_joint": 0.3,
    ".*_ankle_pitch_joint": -0.2,
    ".*_shoulder_pitch_joint": 0.35,
    ".*_elbow_joint": 0.87,
    "left_shoulder_roll_joint": 0.18,
    "right_shoulder_roll_joint": -0.18,
  },
  joint_vel={".*": 0.0},
)

KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.78),
  joint_pos={
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
  },
  joint_vel={".*": 0.0},
)

PARKOUR_DEBUG_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.9),
  joint_pos={
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

# This enables all collisions, including self collisions.
# Self-collisions are given condim=1 while foot collisions
# are given condim=3.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

PARKOUR_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r"^(?:robot/)?(left|right)_foot_collision_[1-7]$": 3, ".*_collision": 1},
  priority={r"^(?:robot/)?(left|right)_foot_collision_[1-7]$": 1},
  friction={r"^(?:robot/)?(left|right)_foot_collision_[1-7]$": (1.0,)},
  disable_other_geoms=False,
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

# This disables all collisions except the feet.
# Feet get condim=3, all other geoms are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot[1-7]_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Final config.
##

G1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    G1_ACTUATOR_5020,
    G1_ACTUATOR_7520_14,
    G1_ACTUATOR_7520_22,
    G1_ACTUATOR_4010,
    G1_ACTUATOR_WAIST,
    G1_ACTUATOR_ANKLE,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_g1_robot_cfg() -> EntityCfg:
  """Get a fresh G1 robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=G1_ARTICULATION,
  )


def get_g1_parkour_robot_cfg() -> EntityCfg:
  """Get the torso-root G1 parkour robot for flat MuJoCo debug play."""
  return EntityCfg(
    init_state=PARKOUR_DEBUG_KEYFRAME,
    collisions=(PARKOUR_COLLISION,),
    spec_fn=get_g1_parkour_flat_debug_spec,
    articulation=G1_ARTICULATION,
  )


def get_g1_parkour_obstacle_robot_cfg() -> EntityCfg:
  """Get the parkour robot with deterministic low-block/gap debug geoms."""
  return EntityCfg(
    init_state=PARKOUR_DEBUG_KEYFRAME,
    collisions=(PARKOUR_COLLISION,),
    spec_fn=get_g1_parkour_obstacle_debug_spec,
    articulation=G1_ARTICULATION,
  )


def get_g1_parkour_complex_terrain_robot_cfg() -> EntityCfg:
  """Get the parkour robot with deterministic complex-terrain debug geoms."""
  return EntityCfg(
    init_state=PARKOUR_DEBUG_KEYFRAME,
    collisions=(PARKOUR_COLLISION,),
    spec_fn=get_g1_parkour_complex_terrain_debug_spec,
    articulation=G1_ARTICULATION,
  )


G1_ACTION_SCALE: dict[str, float] = {}
for a in G1_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    G1_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_g1_robot_cfg())

  viewer.launch(robot.spec.compile())
