from dataclasses import dataclass

from mjlab.tasks.velocity.config.open_duck_mini_v2.rough_env_cfg import (
  OpenDuckMiniV2RoughEnvCfg,
)


@dataclass
class OpenDuckMiniV2FlatEnvCfg(OpenDuckMiniV2RoughEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    assert self.scene.terrain is not None
    self.scene.terrain.terrain_type = "plane"
    self.scene.terrain.terrain_generator = None
    self.curriculum.terrain_levels = None

    self.curriculum.command_vel = None

    assert self.events.push_robot is not None
    self.events.push_robot.params["velocity_range"] = {
      "x": (-0.5, 0.5),
      "y": (-0.5, 0.5),
    }


@dataclass
class OpenDuckMiniV2FlatEnvCfg_PLAY(OpenDuckMiniV2FlatEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    # Effectively infinite episode length.
    self.episode_length_s = int(1e9)
