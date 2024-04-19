import unittest

from pandas.core.tools.numeric import pd
from stable_baselines3.common.env_checker import check_env

from .env import BatteryGym


class TestBGTEnv(unittest.TestCase):
    def test_env(self):
        df_path = "./bot/data/april15-may7_2023.csv"
        df = pd.read_csv(df_path)
        env = BatteryGym(df)
        check_env(env)
