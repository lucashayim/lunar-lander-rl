"""Treina um agente no LunarLander-v3 com PPO ou DQN (Stable-Baselines3).

Uso:
    python -m src.train --algo ppo --steps 1000000
    python -m src.train --algo dqn --steps 1000000

O modelo é salvo em models/<algo>_lunar_lander.zip.
Hiperparâmetros: padrão do Stable-Baselines3 (os mesmos dos modelos de referência).
"""
import argparse
import os

import gymnasium as gym
from stable_baselines3 import DQN, PPO

ALGOS = {"ppo": PPO, "dqn": DQN}


def main():
    parser = argparse.ArgumentParser(description="Treino no LunarLander-v3")
    parser.add_argument("--algo", choices=ALGOS, default="ppo")
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)
    env = gym.make("LunarLander-v3")
    model = ALGOS[args.algo]("MlpPolicy", env, verbose=1, seed=args.seed)
    model.learn(total_timesteps=args.steps)

    path = f"models/{args.algo}_lunar_lander"
    model.save(path)
    print(f"Modelo salvo em {path}.zip")
    env.close()


if __name__ == "__main__":
    main()
