import pandas as pd
from collections import deque
import numpy as np
from policies.policy import Policy

# out of the box provided by hackathon - uses moving average to decide on actions

class AugmentedMovingAveragePolicy(Policy):
    def __init__(self, window_size=500):
        """
        Constructor for the MovingAveragePolicy.

        :param window_size: The number of past market prices to consider for the moving average (default: 5).
        """
        super().__init__()
        self.window_size = window_size
        self.price_history = deque(maxlen=window_size)

    def act(self, external_state, internal_state):
        market_price = external_state['price']
        self.price_history.append(market_price)

        if float(market_price) < 0:
            charge_kW = internal_state['max_charge_rate']
        elif len(self.price_history) == self.window_size:
            moving_average = np.mean(self.price_history)
            
            if market_price > moving_average:
                charge_kW = -internal_state['max_charge_rate']
            else:
                charge_kW = internal_state['max_charge_rate']
        else:
            charge_kW = 0

        return 0, charge_kW

    def load_historical(self, external_states: pd.DataFrame):   
        for price in external_states['price'].values:
            self.price_history.append(price)
