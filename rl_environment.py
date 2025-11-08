import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    """Reinforcement Learning Trading Environment with random start, hold positions, and scaled rewards."""
    metadata = {"render_modes": ["human"]}

    def __init__(self, df, initial_balance=100000):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = 0  # +1 = long, -1 = short, 0 = flat
        self.current_step = 0

        self.window = 10
        self.features = ['close', 'ema_5', 'ema_15', 'total_OI_change']

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window * len(self.features),),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # 0=Hold, 1=Buy, 2=Sell

    def _get_obs(self):
        start = max(0, self.current_step - self.window)
        obs = self.df[self.features].iloc[start:self.current_step].values
        if len(obs) < self.window:
            pad = np.zeros((self.window - len(obs), len(self.features)))
            obs = np.vstack([pad, obs])
        return obs.flatten().astype(np.float32)

    def step(self, action):
        if self.current_step >= len(self.df) - 1:
            return self._get_obs(), 0.0, True, False, {}

        prev_price = self.df['close'].iloc[self.current_step]
        self.current_step += 1
        curr_price = self.df['close'].iloc[self.current_step]

        # Update position based on action
        if action == 1:  # BUY
            self.position = 1
        elif action == 2:  # SELL
            self.position = -1
        else:  # HOLD
            pass

        # Compute reward based on current position and price change
        reward = (curr_price - prev_price) * self.position

        # Reward shaping (encourage profitable moves)
        reward = reward * (1.5 if reward > 0 else 0.5)

        self.balance += reward
        done = self.current_step >= len(self.df) - 1
        return self._get_obs(), float(reward), done, False, {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.position = 0
        # Randomize start step per episode
        self.current_step = np.random.randint(self.window, len(self.df) - self.window - 1)
        return self._get_obs(), {}

    def render(self):
        print(f"Step {self.current_step} | Balance: {self.balance:.2f} | Position: {self.position}")
