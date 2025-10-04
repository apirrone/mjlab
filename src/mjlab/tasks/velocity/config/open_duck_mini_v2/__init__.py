import gymnasium as gym

gym.register(
  id="Mjlab-Velocity-Rough-Open-Duck-Mini-V2",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.rough_env_cfg:OpenDuckMiniV2RoughEnvCfg",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:OpenDuckMiniV2PPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-Velocity-Rough-Open-Duck-Mini-V2-Play",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.rough_env_cfg:OpenDuckMiniV2RoughEnvCfg_PLAY",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:OpenDuckMiniV2PPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-Velocity-Flat-Open-Duck-Mini-V2",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.flat_env_cfg:OpenDuckMiniV2FlatEnvCfg",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:OpenDuckMiniV2PPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-Velocity-Flat-Open-Duck-Mini-V2-Play",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.flat_env_cfg:OpenDuckMiniV2FlatEnvCfg_PLAY",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:OpenDuckMiniV2PPORunnerCfg",
  },
)

# gym.register(
#   id="Mjlab-Velocity-Rough-Unitree-G1",
#   entry_point="mjlab.envs:ManagerBasedRlEnv",
#   disable_env_checker=True,
#   kwargs={
#     "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeG1RoughEnvCfg",
#     "rl_cfg_entry_point": f"{__name__}.rl_cfg:UnitreeG1PPORunnerCfg",
#   },
# )

# gym.register(
#   id="Mjlab-Velocity-Rough-Unitree-G1-Play",
#   entry_point="mjlab.envs:ManagerBasedRlEnv",
#   disable_env_checker=True,
#   kwargs={
#     "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeG1RoughEnvCfg_PLAY",
#     "rl_cfg_entry_point": f"{__name__}.rl_cfg:UnitreeG1PPORunnerCfg",
#   },
# )

# gym.register(
#   id="Mjlab-Velocity-Flat-Unitree-G1",
#   entry_point="mjlab.envs:ManagerBasedRlEnv",
#   disable_env_checker=True,
#   kwargs={
#     "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeG1FlatEnvCfg",
#     "rl_cfg_entry_point": f"{__name__}.rl_cfg:UnitreeG1PPORunnerCfg",
#   },
# )

# gym.register(
#   id="Mjlab-Velocity-Flat-Unitree-G1-Play",
#   entry_point="mjlab.envs:ManagerBasedRlEnv",
#   disable_env_checker=True,
#   kwargs={
#     "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeG1FlatEnvCfg_PLAY",
#     "rl_cfg_entry_point": f"{__name__}.rl_cfg:UnitreeG1PPORunnerCfg",
#   },
# )
