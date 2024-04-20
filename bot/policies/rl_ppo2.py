import os
import sys

import numpy as np
import pandas as pd
from stable_baselines3 import PPO


PATH = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from BGT.env import P_mean, P_std, pv_max

from .policy import Policy


class RlPpo2Policy(Policy):
    def __init__(self, window_size=5):
        """
        Constructor for the MovingAveragePolicy.

        :param window_size: The number of past market prices to consider for the moving average (default: 5).
        """
        super().__init__()
        self.window_size = window_size
        self.price_history = []

        self.rlpolicy = PPO.load(os.path.join(PATH, "models/spot_soc_pv/PPOv2_step29M"))
        self.capacity_kWh = 13.0
        self.max_charge_rate = 5.0

    def act(self, external_state, internal_state):
        market_price = external_state["price"]
        pv_power = external_state["pv_power"]
        # p_mean, p_std = calculate_mean_std(self)
        p = (market_price - P_mean) / 3 * P_std
        soc = internal_state["battery_soc"] / self.capacity_kWh
        pv = pv_power / pv_max
        if p > 1:
            p = 1
        if p < -1:
            p = -1
        if soc > 1:
            soc = 1
        if soc < 0:
            soc = 0
        if pv > 1:
            pv = 1

        observation = np.array([p, soc, pv], dtype=np.float32)
        action = self.rlpolicy.predict(observation)
        charge_kW = action[0][1] * self.max_charge_rate
        solar_kW_to_battery = action[0][0] * self.max_charge_rate

        return solar_kW_to_battery, charge_kW

    def load_historical(self, external_states: pd.DataFrame):
        for price in external_states["price"].values:
            self.price_history.append(price)
