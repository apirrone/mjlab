"""Open Duck Mini V2 constants."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.entity import EntityCfg
from mjlab.utils.os import update_assets

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
  joint_vel={".*": 0.0},
)

OPEN_DUCK_MINI_V2_ROBOT_CFG = EntityCfg(
  init_state=KNEES_BENT_KEYFRAME,
  # collisions=(FULL_COLLISION,),
  spec_fn=get_spec,
  # articulation=G1_ARTICULATION,
)

OPEN_DUCK_MINI_V2_ACTION_SCALE: dict[str, float] = {}
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
