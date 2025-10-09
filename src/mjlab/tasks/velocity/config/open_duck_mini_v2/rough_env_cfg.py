import math
from dataclasses import dataclass, replace

from mjlab.asset_zoo.robots.open_duck_mini_v2.open_duck_mini_v2_constants import (
  OPEN_DUCK_MINI_V2_ACTION_SCALE,
  OPEN_DUCK_MINI_V2_ROBOT_CFG,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.config.open_duck_mini_v2.poly_reference_motion import (
  PolyReferenceMotion,
)
from mjlab.tasks.velocity.velocity_env_cfg import LocomotionNoLinVelVelocityEnvCfg
from mjlab.utils.spec_config import ContactSensorCfg


@dataclass
class OpenDuckMiniV2RoughEnvCfg(LocomotionNoLinVelVelocityEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    foot_contact_sensors = [
      ContactSensorCfg(
        name="left_foot_ground_contact",
        body1="foot_assembly",
        body2="terrain",
        num=1,
        data=("found",),
        reduce="netforce",
      ),
      ContactSensorCfg(
        name="right_foot_ground_contact",
        body1="foot_assembly_2",
        body2="terrain",
        num=1,
        data=("found",),
        reduce="netforce",
      ),
    ]
    open_duck_mini_v2_cfg = replace(
      OPEN_DUCK_MINI_V2_ROBOT_CFG, sensors=tuple(foot_contact_sensors)
    )
    self.scene.entities = {"robot": open_duck_mini_v2_cfg}

    sensor_names = ["left_foot_ground_contact", "right_foot_ground_contact"]
    geom_names = ["left_foot_bottom_tpu", "right_foot_bottom_tpu"]

    self.events.foot_friction.params["asset_cfg"].geom_names = geom_names

    self.actions.joint_pos.scale = OPEN_DUCK_MINI_V2_ACTION_SCALE

    self.rewards.air_time.params["sensor_names"] = sensor_names
    self.rewards.action_rate_l2.weight = -0.5

    # TODO tune
    self.rewards.pose.params["std"] = {
      # Lower body.
      r".*hip_pitch.*": 0.3,
      r".*hip_roll.*": 0.15,
      r".*hip_yaw.*": 0.15,
      r".*knee.*": 0.35,
      r".*ankle.*": 0.25,
      # head
      r".*neck.*": 0.15,
      r".*head.*": 0.1,
    }

    self.viewer.body_name = "base"
    self.commands.twist.viz.z_offset = 0.75
    self.commands.twist.ranges = mdp.UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(-0.5, 0.5),
      lin_vel_y=(-0.5, 0.5),
      ang_vel_z=(-1.0, 1.0),
      heading=(-math.pi, math.pi),
    )

    self.curriculum.command_vel = None


@dataclass
class OpenDuckMiniV2RoughEnvCfg_PLAY(OpenDuckMiniV2RoughEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    # Effectively infinite episode length.
    self.episode_length_s = int(1e9)

    if self.scene.terrain is not None:
      if self.scene.terrain.terrain_generator is not None:
        self.scene.terrain.terrain_generator.curriculum = False
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.border_width = 10.0
