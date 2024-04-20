import os
import sys

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


from BGT.envj import BatteryGymJ

MODEL_DIR = "./models/j/"
ALGO = "PPO"
VERSION = "1"
TENSORBOARD_LOG_DIR = "./Battery_tensorboard/"


def main():
    df = get_clean_data()

    env = BatteryGymJ(df)

    tensorboard_log = os.path.join(TENSORBOARD_LOG_DIR, f"{ALGO}v{VERSION}")

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log)
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
    df = create_features(df)

    return df


def create_features(df):
    df["return"] = df["price"].pct_change()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.day_of_week
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["sin_day"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["cos_day"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    # Define peak hours
    df["is_peak"] = df["hour"].apply(lambda x: 1 if 17 <= x < 21 else 0)
    df["is_off_peak"] = df["hour"].apply(
        lambda x: 1 if 0 <= x < 17 or 21 <= x < 24 else 0
    )
    df["price_negative"] = (df["price"] < 0).astype(int)
    return df


if __name__ == "__main__":
    main()
