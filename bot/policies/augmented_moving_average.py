import pandas as pd
from collections import deque
import numpy as np
from policies.policy import Policy
#
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

    def act(self, external_state, internal_state):

        solar_to_battery = 0

        market_price = external_state['price']
        self.price_history.append(market_price)

        if float(market_price) < 0:
            charge_kW = internal_state['max_charge_rate']
            solar_to_battery = int(float(external_state['pv_power']))
        elif len(self.price_history) == self.window_size:
            moving_average = np.mean(self.price_history)
            
            if market_price > moving_average:
                charge_kW = -internal_state['max_charge_rate']
                ## LOW_BATT_THRESHOLD should be very small, around 20% at absolute maximum
                ## CHARGE_SCALE_FACTOR should take into account not being too high to avoid the sales to the grid but not too low such
                ## that we get periods of 0% capacity where we'll miss out on sales when the price is high due to battery being drained
                ## and pv == 0
                #if battery_capacity < LOW_BATT_THRESHOLD:
                #   solar_to_battery = int(CHARGE_SCALE_FACTOR * int(float(external_state['pv_power'])))
            else:
                charge_kW = internal_state['max_charge_rate']
                solar_to_battery = int(0.7 * int(float(external_state['pv_power'])))
        else:
            charge_kW = 0
        

        return solar_to_battery, charge_kW

    def load_historical(self, external_states: pd.DataFrame):   
        for price in external_states['price'].values:
            self.price_history.append(price)
