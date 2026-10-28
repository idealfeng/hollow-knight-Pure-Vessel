"""Evaluate a frozen policy concurrently through the existing worker pool."""


def interrupt_episode_stats(rewards, steps, actions, reward_parts):
    """Discard partial training scores when evaluation starts new fights."""
    interrupted = []
    for index in range(len(steps)):
        if steps[index]:
            interrupted.append({"env": index, "steps": steps[index],
                                "reward": rewards[index],
                                "reward_parts": dict(reward_parts[index])})
        rewards[index] = 0.0
        steps[index] = 0
        actions[index].clear()
        reward_parts[index].clear()
    return interrupted


def evaluate_pool(pool, model, normalize, episodes_per_env, on_result):
    """Run complete greedy fights without adding samples to PPO's buffer.

    Each worker advances independently, including while other workers reset.
    Freeze each finished worker so it stays still until training resumes.
    """
    if episodes_per_env < 1:
        raise ValueError("episodes_per_env must be positive")
    count = len(pool.connections)
    completed = [0] * count
    rewards = [0.0] * count
    steps = [0] * count
    pending = {}
    results = []
    results_for_start = {}

    def send(index, operation, payload=None):
        pool.send(index, operation, payload)
        pending[pool.connections[index]] = (index, operation)

    def step(index, observation):
        action = model.greedy_act(normalize(observation))
        send(index, "step", action)

    for index in range(count):
        send(index, "reset")

    while pending:
        for connection in pool.ready(pending):
            index, operation = pending.pop(connection)
            payload = pool.recv(index)
            if operation == "reset":
                observation, _ = payload
                # Keep the observation until SPEED has been acknowledged.
                results_for_start[index] = observation
                send(index, "start")
            elif operation == "start":
                step(index, results_for_start.pop(index))
            elif operation == "step":
                observation, reward, terminated, truncated, info = payload
                rewards[index] += reward
                steps[index] += 1
                if not (terminated or truncated):
                    step(index, observation)
                    continue
                completed[index] += 1
                result = {"env": index, "episode_in_eval": completed[index],
                          "reward": rewards[index], "steps": steps[index],
                          "info": info}
                results.append(result)
                on_result(result)
                rewards[index], steps[index] = 0.0, 0
                if completed[index] < episodes_per_env:
                    send(index, "reset")
                else:
                    send(index, "freeze")
            # A freeze acknowledgment removes the final pending operation.
    return results
