import os
import sys

import pandas as pd
from stable_baselines3 import PPO

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


from BGT.env import BatteryGym

MODEL_DIR = "./models/spot_soc_pv/"
ALGO = "PPO"
VERSION = "2"
TENSORBOARD_LOG_DIR = "./Battery_tensorboard/"


def main():
    df = get_clean_data()

    env = BatteryGym(df)

    tensorboard_log = os.path.join(TENSORBOARD_LOG_DIR, f"{ALGO}v{VERSION}")

    model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=tensorboard_log)
    time_steps = 1_000_000
    for i in range(1, 10000):
        if i % 100 == 0:
            steps = f"{i}M"
            model_name = f"{ALGO}v{VERSION}_step{steps}"

            model.learn(
                total_timesteps=time_steps,
                tb_log_name=model_name,
                reset_num_timesteps=False,
            )
            model.save(f"{MODEL_DIR}{model_name}")


def get_clean_data():
    col = ["timestamp", "price", "demand", "pv_power"]
    df = pd.read_csv("./bot/data/training_data.csv")
    df = df[col]
    df.dropna(inplace=True)

    return df


if __name__ == "__main__":
    main()
