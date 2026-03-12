from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg


@configclass
class G1InspireFlatDefaultPPORunnerCfg(G1FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "g1_inspire_flat_default"
        self.run_name = "default"
        self.max_iterations = 1500
