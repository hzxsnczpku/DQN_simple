import os
import click
from train import train
from wrapper import Atari_Wrapper
from estimator import DQN


@click.command()
@click.option('--game_name', prompt='game name:')
def main(game_name):
    assert 'NoFrameskip-v4' in game_name

    env = Atari_Wrapper(game_name)
    estimator = DQN(env.action_n, 1e-4, 0.99)
    base_path = os.path.join(os.getcwd(), 'train_log', game_name[:-14])

    train(env,
          estimator,
          base_path,
          batch_size=32,
          epsilon=0.01,
          update_every=4,
          update_target_every=2500,
          learning_starts=5000,
          memory_size=100000,
          num_iterations=200000000)


if __name__ == "__main__":
    main()
