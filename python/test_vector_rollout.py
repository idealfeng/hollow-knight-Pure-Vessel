"""Offline checks that multi-environment GAE never crosses env boundaries."""

import numpy as np

from rollout_buffer import VectorRolloutBuffer


buffer = VectorRolloutBuffer(
    steps_per_env=2, num_envs=2, state_size=1, num_action_heads=1
)

# env 0: rewards 1,1 and then bootstrap with V=10 -> advantages 12,11.
buffer.add(0, [0], [0], [0.0], 0.0, 1.0, False)
buffer.add(0, [1], [0], [0.0], 0.0, 1.0, False)

# env 1 terminates on its second transition, so its huge supplied bootstrap
# value must be ignored -> advantages 4,2 rather than leaking across envs.
buffer.add(1, [2], [0], [0.0], 0.0, 2.0, False)
buffer.add(1, [3], [0], [0.0], 0.0, 2.0, True)

obs, actions, logprobs, advantages, returns, values = buffer.compute_gae(
    [10.0, 999.0], gamma=1.0, gae_lambda=1.0
)

assert buffer.is_full
assert obs.shape == (4, 1)
assert actions.shape == (4, 1)
assert logprobs.shape == (4, 1)
assert np.allclose(advantages.numpy(), [12.0, 11.0, 4.0, 2.0])
assert np.allclose(returns.numpy(), advantages.numpy())
assert np.allclose(values.numpy(), 0.0)
print("vector rollout GAE: PASS")
