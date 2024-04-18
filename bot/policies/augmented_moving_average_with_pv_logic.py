import pandas as pd
from collections import deque
import numpy as np
from policies.policy import Policy
#
# out of the box provided by hackathon - uses moving average to decide on actions

class PV1AugmentedMovingAveragePolicy(Policy):
    def __init__(self, window_size=250,low_batt_threshold=0, charge_scale_factor=1, discharge_scale_factor=1):
        """
        Constructor for the MovingAveragePolicy.

        :param window_size: The number of past market prices to consider for the moving average (default: 5).
        """
        super().__init__()
        self.window_size = int(window_size)
        self.price_history = deque(maxlen=int(window_size))
        self.battery_capacity_kwh = 13
        self.low_batt_threshold = low_batt_threshold  # As percentage of total capacity
        self.charge_scale_factor = charge_scale_factor
        self.discharge_scale_factor = discharge_scale_factor  # Scale factor for grid discharging
       
    def energy_to_power_rate(self, energy_kWh: float) -> float:
        """
        Convert energy in kilowatt-hours (kWh) to power in kilowatts (kW) based on a given time step duration.

        :param energy_kWh: The amount of energy in kilowatt-hours (kWh).
        :param time_step_duration_minutes: The duration of the time step in minutes.
        :return: The equivalent power in kilowatts (kW).
        """
        time_step_duration_hours = 5.0 / 60.0  # Convert minutes to hours
        power_kW = energy_kWh / time_step_duration_hours
        return power_kW
        
    def act(self, external_state, internal_state):
        market_price = external_state['price']
        pv_power = float(external_state['pv_power'])
        max_charge_rate = internal_state['max_charge_rate']
        battery_soc = float(internal_state["battery_soc"])

        charge_kW = 0
        solar_to_battery = 0

        self.price_history.append(market_price)

        if float(market_price) < 0:
            # charge_kW = internal_state['max_charge_rate']
            # solar_to_battery = int(float(pv_power))
             
            # Calculate the maximum energy that can be added to the battery in this time step (in kWh)
            max_energy_addition_kWh = self.battery_capacity_kwh - battery_soc
            max_charge_power_kW_based_on_capacity = self.energy_to_power_rate(max_energy_addition_kWh)

            # Determine the actual charging power by considering both the max charge rate and the capacity limit
            actual_charge_kW = min(max_charge_rate, max_charge_power_kW_based_on_capacity)

            # Determine how much of the available solar power to direct to the battery
            # Ensure it does not exceed the calculated actual charging power
            solar_to_battery = min(float(pv_power), actual_charge_kW*self.charge_scale_factor)
            
            
        elif len(self.price_history) == self.window_size:
            moving_average = np.mean(self.price_history)
            
            if market_price > moving_average:
                # charge_kW = -max_charge_rate
                ## LOW_BATT_THRESHOLD should be very small, around 20% at absolute maximum
                ## CHARGE_SCALE_FACTOR should take into account not being too high to avoid the sales to the grid but not too low such
                ## that we get periods of 0% capacity where we'll miss out on sales when the price is high due to battery being drained
                ## and pv == 0
                # Prices are high: consider discharging
                if float(battery_soc)/self.battery_capacity_kwh > self.low_batt_threshold:
                    max_discharge_power_kW = self.energy_to_power_rate(battery_soc)
                    charge_kW = -min(max_charge_rate * self.discharge_scale_factor, max_discharge_power_kW)
                    
                if (float(battery_soc)/self.battery_capacity_kwh) < self.low_batt_threshold:
                    solar_to_battery = int(self.charge_scale_factor * int(float(pv_power)))
                #if battery_capacity < LOW_BATT_THRESHOLD:
                #   solar_to_battery = int(CHARGE_SCALE_FACTOR * int(float(external_state['pv_power'])))
            else:
                # charge_kW = max_charge_rate
                # solar_to_battery = int(0.7 * int(float(external_state['pv_power'])))
                # we're potentially wasting excess charge here that could go to the grid -> we will get extra $$ usually in this case becuase it's off peak time
                max_energy_addition_kWh = self.battery_capacity_kwh - battery_soc
                max_charge_power_kW_based_on_capacity = self.energy_to_power_rate(max_energy_addition_kWh)
                charge_kW = min(max_charge_rate * self.charge_scale_factor, max_charge_power_kW_based_on_capacity) 
                solar_to_battery = min(float(pv_power), charge_kW)           
        else:
            charge_kW = 0
            solar_to_battery = 0
        
        return int(solar_to_battery), int(charge_kW)

    def load_historical(self, external_states: pd.DataFrame):   
        for price in external_states['price'].values:
            self.price_history.append(price)