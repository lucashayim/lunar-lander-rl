"""Assiste ao agente treinado pousando (abre uma janela) ou grava vídeos.

Uso:
    python -m src.replay --algo ppo                 # janela com 10 pousos
    python -m src.replay --algo ppo --record 5      # grava 5 episódios em videos/
"""
import argparse

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN, PPO

ALGOS = {"ppo": PPO, "dqn": DQN}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=ALGOS, default="ppo")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--record", type=int, default=0, help="grava N episódios em videos/")
    args = parser.parse_args()

    if args.record:
        env = gym.make("LunarLander-v3", render_mode="rgb_array")
        env = RecordVideo(env, video_folder="videos/", episode_trigger=lambda e: True)
        episodes = args.record
    else:
        env = gym.make("LunarLander-v3", render_mode="human")
        episodes = args.episodes

    model = ALGOS[args.algo].load(f"models/{args.algo}_lunar_lander.zip", device="cpu",
                                  custom_objects={"lr_schedule": lambda _: 0.0,
                                                  "exploration_schedule": lambda _: 0.0,
                                                  "clip_range": lambda _: 0.2})

    for ep in range(episodes):
        obs, _ = env.reset()
        done, score = False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            score += reward
            done = terminated or truncated
        print(f"Episódio {ep + 1}: pontuação {score:.1f}")
    env.close()


if __name__ == "__main__":
    main()
