from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.tasks.velocity.config.open_duck_mini_v2.poly_reference_motion import (
  PolyReferenceMotion,
)


class PRMManager(ManagerTermBase):
  def __init__(self, env: ManagerBasedRlEnv):
    super().__init__(env)
    self.PRM = PolyReferenceMotion(
      "/home/antoine/MISC/mjlab/src/mjlab/tasks/velocity/config/open_duck_mini_v2/polynomial_coefficients.pkl"
    )
    self.imitation_i: int = 0

  def tick(self):
    self.imitation_i = (self.imitation_i + 1) % self.PRM.nb_steps_in_period

  def get_current_reference_motion(self, dx, dy, dtheta):
    return self.PRM.get_reference_motion(dx, dy, dtheta, self.imitation_i)
