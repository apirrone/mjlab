import gymnasium as gym

# TODO add rough back
gym.register(
  id="Mjlab-Velocity-Flat-Open-Duck-Mini-V2",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  # entry_point="mjlab.tasks.velocity.config.open_duck_mini_v2.imitation_env:ImitationEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.flat_env_cfg:OpenDuckMiniV2FlatEnvCfg",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:OpenDuckMiniV2PPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-Velocity-Flat-Open-Duck-Mini-V2-Play",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  # entry_point="mjlab.tasks.velocity.config.open_duck_mini_v2.imitation_env:ImitationEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.flat_env_cfg:OpenDuckMiniV2FlatEnvCfg_PLAY",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:OpenDuckMiniV2PPORunnerCfg",
  },
)
