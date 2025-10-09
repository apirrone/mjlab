import torch

from mjlab.envs import types
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.tasks.velocity.config.open_duck_mini_v2.prm_manager import PRMManager


class ImitationEnv(ManagerBasedRlEnv):
    def __init__(
        self,
        cfg: ManagerBasedRlEnvCfg,
        device: str,
        render_mode: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(cfg, device, render_mode, **kwargs)

        self.prm_manager: PRMManager = PRMManager(self)

    def step(self, action: torch.Tensor) -> types.VecEnvStepReturn:
        self.prm_manager.tick()
        return super().step(action)
