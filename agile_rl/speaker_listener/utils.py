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






"""
multiprocessing_env
Author: Burak M Gonultas
https://github.com/gonultasbu
---
Original Reference:
https://github.com/Farama-Foundation/SuperSuit/issues/43#issuecomment-751792111
"""
from multiprocessing import Pipe, Process

import numpy as np


def worker(remote, parent_remote, env_fn_wrapper):
    """Worker class

    References:
        https://github.com/openai/baselines/tree/master/baselines/common/vec_env
    """
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            data = {
                possible_agent: np.array(data[idx]).squeeze()
                for idx, possible_agent in enumerate(env.possible_agents)
            }
            ob, reward, dones, truncs, info = env.step(data)
            ob = list(ob.values())
            reward = list(reward.values())
            dones = list(dones.values())
            info = list(info.values())
            truncs = list(truncs.values())
            remote.send((ob, reward, dones, truncs, info))
        elif cmd == "reset":
            ob, infos = env.reset(seed=data, options=None)
            ob = list(ob.values())
            infos = list(infos.values())
            remote.send((ob, infos))
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "seed":
            env.seed(data)
        elif cmd == "render":
            env.render()
        else:
            raise NotImplementedError


class VecEnv:
    """An abstract asynchronous, vectorized environment

    References:
        https://github.com/openai/baselines/tree/master/baselines/common/vec_env
    """

    def __init__(self, num_envs, possible_agents):
        self.num_envs = num_envs
        self.agents = possible_agents
        self.num_agents = len(self.agents)

    def reset(self, seed=None, options=None):
        """
        Reset all the environments and return an array of
        observations, or a tuple of observation arrays.
        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.
        You should not call this if a step_async run is
        already pending.
        """
        pass

    def step_wait(self):
        """
        Wait for the step taken with step_async().
        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a tuple of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close(self):
        """
        Clean up the environments' resources.
        """
        pass

    def step(self, actions):
        passed_actions_list = [[] for _ in list(actions.values())[0]]
        for env_idx, _ in enumerate(list(actions.values())[0]):
            for agent_idx, possible_agent in enumerate(self.agents):
                passed_actions_list[env_idx].append(actions[possible_agent][agent_idx])
        self.step_async(passed_actions_list)
        return self.step_wait()


class CloudpickleWrapper:
    """Uses cloudpickle to serialize contents
    (otherwise multiprocessing tries to use pickle)

    References:
        https://github.com/openai/baselines/tree/master/baselines/common/vec_env
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle

        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle

        self.x = pickle.loads(ob)


class SubprocVecEnv(VecEnv):
    """Vectorized environment class that collects samples in parallel using subprocesses

    Args:
        env_fns (list): list of gym environments to run in subprocesses

    References:
        https://github.com/openai/baselines/tree/master/baselines/common/vec_env
    """

    def __init__(self, env_fns):
        self.env = env_fns[0]()
        self.waiting = False
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.closed = False
        self.nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.nenvs)])
        self.ps = [
            Process(
                target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn))
            )
            for (work_remote, remote, env_fn) in zip(
                self.work_remotes, self.remotes, env_fns
            )
        ]
        for p in self.ps:
            p.daemon = (
                True  # If the main process crashes, we should not cause things to hang
            )
            p.start()
        for remote in self.work_remotes:
            remote.close()
        VecEnv.__init__(
            self,
            len(env_fns),
            self.env.possible_agents,
        )

    def seed(self, value):
        for i_remote, remote in enumerate(self.remotes):
            remote.send(("seed", value + i_remote))

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, truncs, infos = zip(*results)

        ret_obs_dict = {
            possible_agent: []
            for idx, possible_agent in enumerate(self.env.possible_agents)
        }
        ret_rews_dict = {
            possible_agent: []
            for idx, possible_agent in enumerate(self.env.possible_agents)
        }
        ret_dones_dict = {
            possible_agent: []
            for idx, possible_agent in enumerate(self.env.possible_agents)
        }
        ret_truncs_dict = {
            possible_agent: []
            for idx, possible_agent in enumerate(self.env.possible_agents)
        }
        ret_infos_dict = {
            possible_agent: []
            for idx, possible_agent in enumerate(self.env.possible_agents)
        }
        for env_idx, _ in enumerate(obs):
            for agent_idx, possible_agent in enumerate(self.env.possible_agents):
                ret_obs_dict[possible_agent].append(obs[env_idx][agent_idx])
                ret_rews_dict[possible_agent].append(rews[env_idx][agent_idx])
                ret_dones_dict[possible_agent].append(dones[env_idx][agent_idx])
                ret_truncs_dict[possible_agent].append(truncs[env_idx][agent_idx])
                ret_infos_dict[possible_agent].append(infos[env_idx][agent_idx])

        for agent_idx, possible_agent in enumerate(self.env.possible_agents):
            for op_dict in [
                ret_obs_dict,
                ret_rews_dict,
                ret_dones_dict,
                ret_truncs_dict,
                ret_infos_dict,
            ]:
                op_dict[possible_agent] = np.stack(op_dict[possible_agent])
        return (
            ret_obs_dict,
            ret_rews_dict,
            ret_dones_dict,
            ret_truncs_dict,
            ret_infos_dict,
        )

    def reset(self, seed=None, options=None):
        for remote in self.remotes:
            remote.send(("reset", seed))
        results = [remote.recv() for remote in self.remotes]
        obs, infos = zip(*results)
        ret_obs_dict = {
            possible_agent: []
            for idx, possible_agent in enumerate(self.env.possible_agents)
        }
        ret_infos_dict = {
            possible_agent: []
            for idx, possible_agent in enumerate(self.env.possible_agents)
        }
        for env_idx, _ in enumerate(obs):
            for agent_idx, possible_agent in enumerate(self.env.possible_agents):
                ret_obs_dict[possible_agent].append(obs[env_idx][agent_idx])
                ret_infos_dict[possible_agent].append(infos[env_idx][agent_idx])
        for agent_idx, possible_agent in enumerate(self.env.possible_agents):
            for op_dict in [
                ret_obs_dict,
                ret_infos_dict,
            ]:
                op_dict[possible_agent] = np.stack(op_dict[possible_agent])
        return (ret_obs_dict, ret_infos_dict)

    def render(self):
        self.remotes[0].send(("render", None))

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.ps:
            p.join()
            self.closed = True

    def sample_personas(self, is_train, is_val=True, path="./"):
        return self.env.sample_personas(is_train=is_train, is_val=is_val, path=path)