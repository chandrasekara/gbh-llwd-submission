import os
import sys

import pandas as pd
from stable_baselines3 import A2C, PPO

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


from BGT.env import BatteryGym


def main():
    df = get_clean_data()

    env = BatteryGym(df)

    # model = A2C("MlpPolicy", env, verbose=1, tensorboard_log="./Battery_tensorboard/")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./Battery_tensorboard/")
    for i in range(1, 10000):
        if i % 100 == 0:
            # print(f"Training run {i}")
            print(f"Training run {i}")
            model.learn(
                total_timesteps=10_000,
                tb_log_name=f"PPO2steps_{i}_10k",
                reset_num_timesteps=False,
            )
            model.save(f"./models/spot_soc_pv/PPO2{i}x100k_steps")


def get_clean_data():
    col = ["timestamp", "price", "demand", "pv_power"]
    df = pd.read_csv("./bot/data/training_data.csv")
    df = df[col]
    df.dropna(inplace=True)

    return df


if __name__ == "__main__":
    main()
