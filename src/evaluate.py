"""Avalia modelos treinados em N episódios (política determinística, seeds 0..N-1).

Uso:
    python -m src.evaluate                  # avalia PPO e DQN em models/
    python -m src.evaluate --episodes 200
Gera assets/resultados.json e assets/distribuicao_recompensas.png.
"""
import argparse
import json
import os

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN, PPO

ALGOS = {"PPO": PPO, "DQN": DQN}
SOLVED = 200  # recompensa a partir da qual o episódio é considerado "resolvido"


def run_episodes(model, n):
    env = gym.make("LunarLander-v3")
    rewards = []
    for ep in range(n):
        obs, _ = env.reset(seed=ep)
        done, total = False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        rewards.append(total)
    env.close()
    return np.array(rewards)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()

    os.makedirs("assets", exist_ok=True)
    summary, all_rewards = {}, {}
    for name, cls in ALGOS.items():
        path = f"models/{name.lower()}_lunar_lander.zip"
        if not os.path.exists(path):
            print(f"[pulando] {path} não encontrado")
            continue
        # custom_objects evita problemas de desserialização entre versões do Python
        model = cls.load(
            path,
            device="cpu",
            custom_objects={"lr_schedule": lambda _: 0.0, "exploration_schedule": lambda _: 0.0,
                            "clip_range": lambda _: 0.2},
        )
        r = run_episodes(model, args.episodes)
        all_rewards[name] = r
        summary[name] = {
            "episodios": args.episodes,
            "recompensa_media": round(float(r.mean()), 1),
            "desvio_padrao": round(float(r.std()), 1),
            "mediana": round(float(np.median(r)), 1),
            "minimo": round(float(r.min()), 1),
            "maximo": round(float(r.max()), 1),
            "pct_episodios_ge_200": round(float((r >= SOLVED).mean() * 100), 1),
            "pct_episodios_ge_100": round(float((r >= 100).mean() * 100), 1),
            "pct_episodios_negativos": round(float((r < 0).mean() * 100), 1),
        }
        print(name, json.dumps(summary[name], ensure_ascii=False))

    with open("assets/resultados.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # gráfico: distribuição das recompensas por episódio
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = {"PPO": "#7C3AED", "DQN": "#0891B2"}
    bins = np.linspace(-150, 320, 36)
    for name, r in all_rewards.items():
        ax.hist(r, bins=bins, alpha=0.65, label=f"{name} (média {r.mean():.0f})", color=colors[name])
    ax.axvline(SOLVED, color="#16A34A", linestyle="--", linewidth=1.5, label="Resolvido (200)")
    ax.set_xlabel("Recompensa por episódio")
    ax.set_ylabel("Nº de episódios")
    ax.set_title(f"LunarLander-v3: distribuição das recompensas ({args.episodes} episódios)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig("assets/distribuicao_recompensas.png", dpi=150)
    print("Gráfico salvo em assets/distribuicao_recompensas.png")


if __name__ == "__main__":
    main()
