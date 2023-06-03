import toml
from argparse import ArgumentParser
from os.path import join
import gym
# import time
# import wandb
from torch.utils.tensorboard import SummaryWriter

from games.carracing import RacingNet, CarRacing
from ppo import PPO

CONFIG_FILE = "config.toml"


def load_config():
    with open(CONFIG_FILE, "r") as f:
        config = toml.load(f)

    return config


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="ckpt")
    parser.add_argument("--num_steps", type=int, default=100_000)
    parser.add_argument("--delay_ms", type=int, default=10)
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    return parser.parse_args()


def main():
    cfg = load_config()
    args = parse_args()
    # run_name = f"{int(time.time())}"

    env = CarRacing(frame_skip=0, frame_stack=4,)
    # env = gym.wrappers.RecordEpisodeStatistics(env)
    # env = gym.wrappers.RecordVideo(env, f"videos/")

    net = RacingNet(env.observation_space.shape, env.action_space.shape)

    # if args.track:
    #     import wandb

    #     wandb.init(
    #         project="CarRacing",
    #         entity=args.wandb_entity,
    #         sync_tensorboard=True,
    #         config=vars(args),
    #         name=run_name,
    #         monitor_gym=True,
    #         save_code=True,
    #     )
    # writer = SummaryWriter(f"runs/{run_name}")
    # writer.add_text(
    #     "hyperparameters",
    #     "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    # )


    ppo = PPO(
        env,
        net,
        lr=cfg["lr"],
        gamma=cfg["gamma"],
        batch_size=cfg["batch_size"],
        gae_lambda=cfg["gae_lambda"],
        clip=cfg["clip"],
        value_coef=cfg["value_coef"],
        entropy_coef=cfg["entropy_coef"],
        epochs_per_step=cfg["epochs_per_step"],
        num_steps=cfg["num_steps"],
        horizon=cfg["horizon"],
        save_dir=cfg["save_dir"],
        save_interval=cfg["save_interval"],
    )

    # ppo.load(args.ckpt)

    for i in range(args.num_steps):
        ppo.collect_trajectory(1, delay_ms=args.delay_ms)

    env.close()


if __name__ == "__main__":
    main()
