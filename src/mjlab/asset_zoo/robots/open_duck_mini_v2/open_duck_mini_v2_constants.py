"""Open Duck Mini V2 constants."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import ElectricActuator
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import ActuatorCfg

##
# MJCF and assets.
##

OPEN_DUCK_MINI_V2_XML: Path = (
  MJLAB_SRC_PATH
  / "asset_zoo"
  / "robots"
  / "open_duck_mini_v2"
  / "xmls"
  / "open_duck_mini_v2.xml"
)
assert OPEN_DUCK_MINI_V2_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, OPEN_DUCK_MINI_V2_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(OPEN_DUCK_MINI_V2_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.15),
  joint_pos={
    "left_hip_yaw": 0,
    "left_hip_roll": 0.05,
    "left_hip_pitch": -0.63,
    "left_knee": 1.37,
    "left_ankle": -0.78,
    "neck_pitch": 0,
    "head_pitch": 0,
    "head_yaw": 0,
    "head_roll": 0,
    "right_hip_yaw": 0,
    "right_hip_roll": -0.05,
    "right_hip_pitch": 0.63,
    "right_knee": 1.37,
    "right_ankle": -0.78,
  },
  joint_vel={
    "left_hip_yaw": 0,
    "left_hip_roll": 0.05,
    "left_hip_pitch": -0.63,
    "left_knee": 1.37,
    "left_ankle": -0.78,
    "neck_pitch": 0,
    "head_pitch": 0,
    "head_yaw": 0,
    "head_roll": 0,
    "right_hip_yaw": 0,
    "right_hip_roll": -0.05,
    "right_hip_pitch": 0.63,
    "right_knee": 1.37,
    "right_ankle": -0.78,
  },
)

# <default class="sts3215">
#   <geom contype="0" conaffinity="0"/>
#   <!-- <joint damping="0.60" frictionloss="0.052" armature="0.028"/>
#   <position kp="17.8" kv="0.0" forcerange="-3.35 3.35"/> -->
#   <joint damping="0.56" frictionloss="0.068" armature="0.027"/>
#   <position kp="17.11" kv="0.0" forcerange="-3.23 3.23"/>
# </default>
ACTUATOR_7_4V_STS3215 = ElectricActuator(
  reflected_inertia=0.028,
  velocity_limit=5.24,
  effort_limit=3.35,
)

OPEN_DUCK_MINI_ACTUATOR_7_4V_STS3215 = ActuatorCfg(
  joint_names_expr=[
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
  ],
  effort_limit=ACTUATOR_7_4V_STS3215.effort_limit,
  armature=ACTUATOR_7_4V_STS3215.reflected_inertia,
  stiffness=17.8,
  damping=0.56,
)

OPEN_DUCK_MINI_V2_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(OPEN_DUCK_MINI_ACTUATOR_7_4V_STS3215,),
  soft_joint_pos_limit_factor=0.9,
)

OPEN_DUCK_MINI_V2_ROBOT_CFG = EntityCfg(
  init_state=KNEES_BENT_KEYFRAME,
  # collisions=(FULL_COLLISION,),
  spec_fn=get_spec,
  # articulation=OPEN_DUCK_MINI_V2_ARTICULATION, # Doesn't seem to be needed if all is defined in the xml
)

OPEN_DUCK_MINI_V2_ACTION_SCALE: dict[str, float] = {
  "left_hip_yaw": 1.0,
  "left_hip_roll": 1.0,
  "left_hip_pitch": 1.0,
  "left_knee": 1.0,
  "left_ankle": 1.0,
  "right_hip_yaw": 1.0,
  "right_hip_roll": 1.0,
  "right_hip_pitch": 1.0,
  "right_knee": 1.0,
  "right_ankle": 1.0,
  "neck_pitch": 1.0,
  "head_pitch": 1.0,
  "head_yaw": 1.0,
  "head_roll": 1.0,
}

# for a in G1_ARTICULATION.actuators:
#   e = a.effort_limit
#   s = a.stiffness
#   names = a.joint_names_expr
#   if not isinstance(e, dict):
#     e = {n: e for n in names}
#   if not isinstance(s, dict):
#     s = {n: s for n in names}
#   for n in names:
#     if n in e and n in s and s[n]:
#       G1_ACTION_SCALE[n] = 0.25 * e[n] / s[n]

if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(OPEN_DUCK_MINI_V2_ROBOT_CFG)

  viewer.launch(robot.spec.compile())
