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


class BatteryGymJ(gym.Env):
    def __init__(self, df):
        super().__init__()

        start_step = 0

        historical_data = df.iloc[:start_step]
        future_data = df.iloc[start_step:]

        self.env = BatteryEnv(
            data=future_data, initial_charge_kWh=7.5, initial_profit=0.0
        )
        # print(df.head())

        self.charge_rate = self.env.battery.max_charge_rate_kW

        self.action_space = spaces.Box(
            -1.0, 1.0, shape=(2,), dtype=np.float32
        )  # action space: [solar_kW_to_battery, charge_kW]

        #   observation space: [price, soc, pv,hour]
        low = np.array([-1.0, 0.0, 0.0, 0.0])
        high = np.array([1.0, 1.0, 1.0, 1.0])
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

        hr = self.external_state["hour"] * (0.04167)
        pv = pv_power / pv_max
        if price > 1:
            price = 1
        if price < -1:
            price = -1
        if soc > 1:
            soc = 1
        if soc < 0:
            soc = 0
        observation = np.array([price, soc, pv, hr], dtype=np.float32)
        return observation, {}

    def step(self, action):

        if self.external_state is None:
            observation = np.array([0, 0, 0, 0], dtype=np.float32)
            return observation, 0, True, True, {}

        if self.internal_state is None:
            observation = np.array([0, 0, 0, 0], dtype=np.float32)
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
            observation = np.array([0, 0, 0, 0.0], dtype=np.float32)
            return observation, 0, True, True, {}

        if self.internal_state is None:
            observation = np.array([0, 0, 0, 0.0], dtype=np.float32)
            return observation, 0, True, True, {}

        self.profits.append(self.internal_state["total_profit"])
        self.socs.append(self.internal_state["battery_soc"])

        price = (self.external_state[PRICE_KEY] - P_mean) / 3 * P_std
        soc = self.internal_state["battery_soc"] / self.env.battery.capacity_kWh
        pv = pv_power / pv_max
        hr = self.external_state["hour"] * (0.04167)
        if price > 1:
            price = 1
        if price < -1:
            price = -1
        if soc > 1:
            soc = 1
        if soc < 0:
            soc = 0
        # reward = self.internal_state["profit_delta"]
        reward = self.calculate_combined_logic_reward(
            action, self.external_state, self.internal_state
        )
        observation = np.array([price, soc, pv, hr], dtype=np.float32)
        return observation, reward, False, False, {}

    def render(self):
        pass

    def close(self):
        pass

    def calculate_combined_logic_reward(self, action, external_state, internal_state):
        reward = 0
        solar_kW_to_battery, charge_kW = action
        price = (external_state["price"] - P_mean) / 3 * P_std
        # pv_power = external_state["pv_power"]
        battery_soc = (
            internal_state["battery_soc"] / self.env.battery.capacity_kWh
        )  # Normalize SOC
        is_peak_hour = external_state["hour"] in [17, 18, 19, 20]  # Define peak hours
        is_off_peak_hour = not is_peak_hour

        # Assuming charge_kW is positive for charging and negative for discharging
        energy_amount = (
            abs(charge_kW) / self.env.battery.capacity_kWh
        )  # Use the absolute value for calculations

        # Charging logic
        if charge_kW > 0:
            if is_off_peak_hour:
                cost = (
                    energy_amount * price * 1.05
                )  # 5% penalty for charging during off-peak hours
            else:
                cost = energy_amount * price * 1.40  # Higher cost during peak hours
            reward -= cost
            if solar_kW_to_battery > energy_amount:
                reward += 0.1 * cost  # Bonus for utilizing solar power over grid power

        # Discharging logic
        elif charge_kW < 0:  # Negative charge_kW indicates discharging
            if is_peak_hour:
                revenue = (
                    energy_amount * price * 1.30
                )  # Bonus for discharging during peak hours
            elif is_off_peak_hour:
                revenue = (
                    energy_amount * price * 0.85
                )  # Reduced revenue for discharging during off-peak hours
            else:
                revenue = energy_amount * price
            reward += revenue

        # Battery efficiency and SOC considerations
        if battery_soc > 0.9:  # Discourage reaching full capacity to avoid overcharging
            reward -= 1
        elif battery_soc < 0.1:  # Discourage depleting the battery to avoid damage
            reward -= 1

        return reward

