import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from maddpg import MADDPG


def makeVectEnvs(env_name, num_envs=1):
    """Returns async-vectorized gym environments.

    :param env_name: Gym environment name
    :type env_name: str
    :param num_envs: Number of vectorized environments, defaults to 1
    :type num_envs: int, optional
    """
    return gym.vector.AsyncVectorEnv(
        [lambda: gym.make(env_name) for i in range(num_envs)]
    )


def makeSkillVectEnvs(env_name, skill, num_envs=1):
    """Returns async-vectorized gym environments.

    :param env_name: Gym environment name
    :type env_name: str
    :param skill: Skill wrapper to apply to environment
    :type skill: agilerl.wrappers.learning.Skill
    :param num_envs: Number of vectorized environments, defaults to 1
    :type num_envs: int, optional
    """
    return gym.vector.AsyncVectorEnv(
        [lambda: skill(gym.make(env_name)) for i in range(num_envs)]
    )


def initialPopulation(
    algo,
    state_dim,
    action_dim,
    one_hot,
    net_config,
    INIT_HP,
    actor_network=None,
    critic_network=None,
    population_size=1,
    device="cpu",
    accelerator=None,
):
    """Returns population of identical agents.

    :param algo: RL algorithm
    :type algo: str
    :param state_dim: State observation dimension
    :type state_dim: int
    :param action_dim: Action dimension
    :type action_dim: int
    :param one_hot: One-hot encoding
    :type one_hot: bool
    :param INIT_HP: Initial hyperparameters
    :type INIT_HP: dict
    :param actor_network: Custom actor network, defaults to None
    :type actor_network: nn.Module, optional
    :param critic_network: Custom critic network, defaults to None
    :type critic_network: nn.Module, optional
    :param population_size: Number of agents in population, defaults to 1
    :type population_size: int, optional
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to 'cpu'
    :type device: str, optional
    :param accelerator: Accelerator for distributed computing, defaults to None
    :type accelerator: accelerate.Accelerator(), optional
    """
    population = []
    if algo == "MADDPG":
        for idx in range(population_size):
            agent = MADDPG(
                state_dims=state_dim,
                action_dims=action_dim,
                one_hot=one_hot,
                n_agents=INIT_HP["N_AGENTS"],
                agent_ids=INIT_HP["AGENT_IDS"],
                index=idx,
                max_action=INIT_HP["MAX_ACTION"],
                min_action=INIT_HP["MIN_ACTION"],
                net_config=net_config,
                batch_size=INIT_HP["BATCH_SIZE"],
                lr=INIT_HP["LR"],
                learn_step=INIT_HP["LEARN_STEP"],
                gamma=INIT_HP["GAMMA"],
                tau=INIT_HP["TAU"],
                discrete_actions=INIT_HP["DISCRETE_ACTIONS"],
                actor_networks=actor_network,
                critic_networks=critic_network,
                device=device,
                accelerator=accelerator,
            )
            population.append(agent)

   
    return population


def calculate_vectorized_scores(
    rewards, terminations, include_unterminated=False, only_first_episode=True
):
    episode_rewards = []
    num_envs, _ = rewards.shape

    for env_index in range(num_envs):
        # Find the indices where episodes terminate for the current environment
        termination_indices = np.where(terminations[env_index] == 1)[0]

        # If no terminations, sum the entire reward array for this environment
        if len(termination_indices) == 0:
            episode_reward = np.sum(rewards[env_index])
            episode_rewards.append(episode_reward)
            continue  # Skip to the next environment

        # Initialize the starting index for segmenting
        start_index = 0

        for termination_index in termination_indices:
            # Sum the rewards for the current episode
            episode_reward = np.sum(
                rewards[env_index, start_index : termination_index + 1]
            )

            # Store the episode reward
            episode_rewards.append(episode_reward)

            # If only the first episode is required, break after processing it
            if only_first_episode:
                break

            # Update the starting index for segmenting
            start_index = termination_index + 1

        # If include_unterminated is True, sum the rewards from the last termination index to the end
        if (
            not only_first_episode
            and include_unterminated
            and start_index < len(rewards[env_index])
        ):
            episode_reward = np.sum(rewards[env_index, start_index:])
            episode_rewards.append(episode_reward)

    return episode_rewards


def printHyperparams(pop):
    """Prints current hyperparameters of agents in a population and their fitnesses.

    :param pop: Population of agents
    :type pop: list[object]
    """

    for agent in pop:
        print(
            "Agent ID: {}    Mean 100 fitness: {:.2f}    lr: {}    Batch Size: {}".format(
                agent.index, np.mean(agent.fitness[-100:]), agent.lr, agent.batch_size
            )
        )


def plotPopulationScore(pop):
    """Plots the fitness scores of agents in a population.

    :param pop: Population of agents
    :type pop: list[object]
    """
    plt.figure()
    for agent in pop:
        scores = agent.fitness
        steps = agent.steps[:-1]
        plt.plot(steps, scores)
    plt.title("Score History - Mutations")
    plt.xlabel("Steps")
    plt.ylim(bottom=-400)
    plt.show()






