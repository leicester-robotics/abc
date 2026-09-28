"""``deploy/deploy_policy.py`` with the rollout swapped for local_robot.policy_rollout.

Same CLI as upstream:
  ROBOT_PROFILE=local_yam_config uv run python -m local_robot.deploy_policy ...
"""

import tyro

from deploy import deploy_policy
from deploy.deploy_config import DeployConfig

_upstream_specs = deploy_policy._real_robot_specs


def _real_robot_specs(cfg, **kwargs):
    specs = _upstream_specs(cfg, **kwargs)
    for spec in specs:
        if spec.target == "deploy.robot.gym.policy_rollout":
            spec.target = "local_robot.policy_rollout"
    return specs


deploy_policy._real_robot_specs = _real_robot_specs


if __name__ == "__main__":
    raise SystemExit(deploy_policy.main(tyro.cli(DeployConfig)))
