import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class FrameSkip(gym.Wrapper):
    def __init__(self, env, skip=2):
        """FrameSkip constructor
        Actions are applied every `skip` frames
        Intermediate frames maintain same action

        Args:
            env (gym.Env): Env to wrap
            skip (int | tuple):
                number of frames to skip (may be range to sample from)
                [low, high)
        """
        super().__init__(env)
        self.skip = skip
        self.count = 0
        self.skip_ = 0
        self.action = None

    def step(self, action):
        obs, reward, done, truncated, info = None, 0.0, None, None, None
        act = np.copy(action)
        if isinstance(self.skip, tuple):
            skip = np.random.randint(self.skip[0], self.skip[1])
        else:
            skip = self.skip
        for _ in range(skip + 1):
            obs, r, done, truncated, info = self.env.step(action)
            reward += r
            action = np.copy(act)
            if done:
                break
        return obs, reward / float(skip + 1.0), done, truncated, info


class F1TenthActionTransform(gym.ActionWrapper):
    def __init__(self, env, vmin=1.0, vmax=8.0, steermax=0.4189):
        super().__init__(env)
        self.vmax = vmax
        self.vmin = vmin
        self.steermax = steermax
        low = np.array([[-1.0, 0.0]]).astype(np.float32)
        high = np.array([[1.0, 1.0]]).astype(np.float32)
        env.action_space = gym.spaces.Box(
            low=low, high=high, shape=(1, 2), dtype=np.float32
        )

    def action(self, action):
        action[0][0] *= self.steermax
        action[0][1] *= self.vmax - self.vmin
        action[0][1] += self.vmin
        return action


class F1TenthObsTransform(gym.ObservationWrapper):
    def __init__(self, env, beam_count=40):
        super(F1TenthObsTransform, self).__init__(env)
        self.beam_count = beam_count
        self.last = np.zeros((self.beam_count,))
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self.beam_count,)
        )
        self.scale = 30

    def observation(self, observation):
        scan: gym.spaces.Box = observation["agent_0"]["scan"]
        scan = scan / self.scale
        scan = np.clip(scan, 0, 1)
        indices = np.linspace(0, 1079, self.beam_count, dtype=int)
        scan = scan[indices]
        return scan


class F1TenthTensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    TODO: Not working currently
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.steps = 0
        self.laptimes = dict()
        self.progress = dict()

    def _on_step(self) -> bool:
        for i, env in enumerate(self.locals["env"].envs):
            progs = env.env.unwrapped.agent_progress
            times = env.env.unwrapped.lap_times
            time = self.laptimes.get(i, np.zeros_like(times))
            self.laptimes[i] = time + times
            prog = self.progress.get(i, np.zeros_like(times))
            self.progress[i] = prog + progs

        self.steps += 1
        return True

    def _on_rollout_end(self) -> None:
        for j, env_times in self.laptimes.items():
            for i, _ in enumerate(env_times):
                self.logger.record(
                    f"rollout/progress_{j}_{i}", self.progress[j][i] / self.steps
                )
                self.logger.record(
                    f"rollout/laptimes_{j}_{i}", self.laptimes[j][i] / self.steps
                )
        self.steps = 0
        self.laptimes = dict()
        self.progress = dict()
