import os
import sys
import unittest

from pandas.core.tools.numeric import pd
from stable_baselines3.common.env_checker import check_env

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from trainRL.trainj import get_clean_data

from .env import BatteryGym
from .envj import BatteryGymJ


class TestBGTEnv(unittest.TestCase):
    def test_env(self):
        df_path = "./bot/data/april15-may7_2023.csv"
        df = pd.read_csv(df_path)
        env = BatteryGym(df)
        check_env(env)

    def test_envj(self):
        # df_path = "./bot/data/training_data.csv"
        # df = pd.read_csv(df_path)
        df = get_clean_data()

        env = BatteryGymJ(df)
        check_env(env)
