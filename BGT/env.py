import os
import sys

import gymnasium as gym
import numpy as np
from gymnasium import spaces

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
# print(sys.path)

from bot.tariff_environment import PRICE_KEY, TIMESTAMP_KEY, BatteryEnv

P_mean = 160.0
P_std = 680.0

pv_max = 10.0


class BatteryGym(gym.Env):
    def __init__(self, df):
        super().__init__()

        start_step = 0

        historical_data = df.iloc[:start_step]
        future_data = df.iloc[start_step:]

        self.env = BatteryEnv(
            data=future_data, initial_charge_kWh=7.5, initial_profit=0.0
        )

        self.charge_rate = self.env.battery.max_charge_rate_kW

        self.action_space = spaces.Box(
            -1.0, 1.0, shape=(2,), dtype=np.float32
        )  # action space: [solar_kW_to_battery, charge_kW]
        low = np.array([-1.0, 0.0, 0.0])
        high = np.array([1.0, 1.0, 1.0])
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

    def reset(self, seed=None, options=None):
        (
            self.profits,
            self.socs,
            self.market_prices,
            self.battery_actions,
            self.solar_actions,
            self.pv_inputs,
            self.timestamps,
        ) = ([], [], [], [], [], [], [])
        self.env.current_step = 0
        self.external_state, self.internal_state = self.env.initial_state()
        price = (self.external_state[PRICE_KEY] - P_mean) / 3 * P_std
        soc = self.internal_state["battery_soc"] / self.env.battery.capacity_kWh
        pv_power = float(self.external_state["pv_power"])
        pv = pv_power / pv_max
        if price > 1:
            price = 1
        if price < -1:
            price = -1
        if soc > 1:
            soc = 1
        if soc < 0:
            soc = 0
        observation = np.array([price, soc, pv], dtype=np.float32)
        return observation, {}

    def step(self, action):

        if self.external_state is None:
            observation = np.array([0, 0, 0], dtype=np.float32)
            return observation, 0, True, True, {}

        if self.internal_state is None:
            observation = np.array([0, 0, 0], dtype=np.float32)
            return observation, 0, True, True, {}

        # print("action", action)
        solar_kW_to_battery, charge_kW = action
        solar_kW_to_battery *= self.charge_rate
        charge_kW *= self.charge_rate
        pv_power = float(self.external_state["pv_power"])

        self.market_prices.append(self.external_state[PRICE_KEY])
        self.timestamps.append(self.external_state[TIMESTAMP_KEY])
        self.battery_actions.append(charge_kW)
        self.solar_actions.append(solar_kW_to_battery)
        self.pv_inputs.append(pv_power)

        self.external_state, self.internal_state = self.env.step(
            charge_kW, solar_kW_to_battery, int(pv_power)
        )
        if self.external_state is None:
            observation = np.array([0, 0, 0], dtype=np.float32)
            return observation, 0, True, True, {}

        if self.internal_state is None:
            observation = np.array([0, 0, 0], dtype=np.float32)
            return observation, 0, True, True, {}

        self.profits.append(self.internal_state["total_profit"])
        self.socs.append(self.internal_state["battery_soc"])

        price = (self.external_state[PRICE_KEY] - P_mean) / 3 * P_std
        soc = self.internal_state["battery_soc"] / self.env.battery.capacity_kWh
        pv = pv_power / pv_max
        if price > 1:
            price = 1
        if price < -1:
            price = -1
        if soc > 1:
            soc = 1
        if soc < 0:
            soc = 0
        reward = self.internal_state["profit_delta"]
        observation = np.array([price, soc, pv], dtype=np.float32)
        return observation, reward, False, False, {}

    def render(self):
        pass

    def close(self):
        pass
