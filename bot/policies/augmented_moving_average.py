import pandas as pd
from collections import deque
import numpy as np
from policies.policy import Policy
##
# out of the box provided by hackathon - uses moving average to decide on actions

class AugmentedMovingAveragePolicy(Policy):
    def __init__(self, window_size=250):
        """
        Constructor for the MovingAveragePolicy.

        :param window_size: The number of past market prices to consider for the moving average (default: 5).
        """
        super().__init__()
        self.window_size = window_size
        self.price_history = deque(maxlen=window_size)
        self.battery_capacity_kwh = 13

    def act(self, external_state, internal_state):

        solar_to_battery = 0
        charge_kW = 0

        fraction_of_max_capacity = float(internal_state["battery_soc"])/13

        market_price = external_state['price']
        self.price_history.append(market_price)

        if float(market_price) < 0:
            charge_kW = internal_state['max_charge_rate']
            solar_to_battery = int(float(external_state['pv_power']))
        elif len(self.price_history) == self.window_size:
            moving_average = np.mean(self.price_history)
            
            if market_price > moving_average:
                if fraction_of_max_capacity > 0.5:
                    charge_kW = -internal_state['max_charge_rate']
                elif fraction_of_max_capacity > 0.3:
                    charge_kW = 0.5 * -internal_state['max_charge_rate']
                elif fraction_of_max_capacity > 0.2:
                    charge_kW = 0.25 * -internal_state['max_charge_rate']
                elif fraction_of_max_capacity > 0.05:
                    charge_kW = 0.05 * -internal_state['max_charge_rate']
                ## LOW_BATT_THRESHOLD should be very small, around 20% at absolute maximum
                ## CHARGE_SCALE_FACTOR should take into account not being too high to avoid the sales to the grid but not too low such
                ## that we get periods of 0% capacity where we'll miss out on sales when the price is high due to battery being drained
                ## and pv == 0
                #if (float(internal_state["battery_soc"])/13) < 0.2:
                #    solar_to_battery = int(0.2 * int(float(external_state['pv_power'])))

                #if battery_capacity < LOW_BATT_THRESHOLD:
                #   solar_to_battery = int(CHARGE_SCALE_FACTOR * int(float(external_state['pv_power'])))
            else:
                charge_kW = internal_state['max_charge_rate']

                new_soc = float(internal_state["battery_soc"]) + (charge_kW)/20
                if new_soc > 13:
                    charge_kW = 0



        else:
            charge_kW = 0
        

        return solar_to_battery, charge_kW

    def load_historical(self, external_states: pd.DataFrame):   
        for price in external_states['price'].values:
            self.price_history.append(price)
