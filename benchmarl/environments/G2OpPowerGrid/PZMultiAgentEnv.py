# %%
import numpy as np
import re
from typing import List, Dict, Any, Optional, Type
import functools
import gymnasium as gym
from gymnasium.spaces import Box
from pettingzoo import ParallelEnv
from lightsim2grid import LightSimBackend
import grid2op
from grid2op.Action import PlayableAction
from grid2op.Reward import LinesCapacityReward, BaseReward
from grid2op.Backend import Backend
from grid2op.Chronics import MultifolderWithCache, Multifolder
from grid2op.gym_compat import GymEnv, BoxGymnasiumObsSpace, BoxGymnasiumActSpace
from .utils import *
# %%

_TIME_SERIE_ID = "time serie id"

try:
    from lightsim2grid import LightSimBackend
    backend_cls = LightSimBackend
except ImportError as exc_:
    from grid2op.Backend import PandaPowerBackend
    backend_cls = PandaPowerBackend

class PZMultiAgentEnv(ParallelEnv):
    metadata = {"render_modes": [], "name": "g2op_power_grid"}
    def __init__(self,
                 env_name: str = "l2rpn_idf_2023",
                 zone_names: List[str] = [f"Zone{j}" for j in range(11)],
                 backend_cls: Type[Backend] = backend_cls,
                 use_global_obs: bool = False,
                 use_redispatching_agent: bool = True,
                 add_local_redisp: bool = False,
                 env_g2op_config: Dict[str, Any] = {},
                 local_rewards: Optional[Dict[str, Type[BaseReward]]] = None,
                 shuffle_chronics: bool = True,
                 regex_filter_chronics: Optional[str] = None,
                 zones_path: Optional[str] = None,
                 ):
        
        if use_redispatching_agent and add_local_redisp:
            raise ValueError("""use_redispatching_agent and add_local_redisp are both set to True but only one them is possible.""")

        # Zones definition
        self.zone_names = np.sort(zone_names)

        full_zones_dict = get_zone_dict_from_path(zones_path=zones_path)

        for zone_name in self.zone_names:
            if zone_name not in ZONES_DICT.keys():
                raise ValueError(f"Zone {zone_name} not found in {zones_path if zones_path else "zones_definitions.json"}. Possible zones are {list(full_zones_dict.keys())}")
            
        # We only keep the zones we are interested in
        self.zones_dict = {zone_name: full_zones_dict[zone_name] for zone_name in self.zone_names}
        self.zones_dict = add_missing_keys(self.zones_dict)

        self.use_global_obs = use_global_obs
        self._local_rewards = local_rewards
        self._shuffle_chronics = shuffle_chronics
        self.render_mode = None # We don't need it but it's required by the ParallelEnv class

        self.use_redispatching_agent = use_redispatching_agent
        self.add_local_redisp = add_local_redisp
        self.possible_agents = [f"agent_{i}" for i in range(1, len(self.zone_names) + 1)]
        if self.use_redispatching_agent:
            self.possible_agents = ["redispatching_agent"] + self.possible_agents
        self.agents = self.possible_agents
        

        # Create the grid2op environment
        if regex_filter_chronics is not None and "chronics_class" not in env_g2op_config: 
            env_g2op_config["chronics_class"] = MultifolderWithCache 
        self._update_env_g2op_config_for_rewards(env_g2op_config)
        env = grid2op.make(env_name, 
                           action_class=PlayableAction, 
                           backend=backend_cls(), 
                           **env_g2op_config)
        self.env_g2op = env

        ## Filter chronics
        if regex_filter_chronics is not None:
            compiled_pattern = re.compile(regex_filter_chronics)
            self.env_g2op.chronics_handler.real_data.set_filter(lambda chronic_name: bool(compiled_pattern.match(chronic_name)))
        if regex_filter_chronics is not None or \
            type(self.env_g2op.chronics_handler.real_data) == MultifolderWithCache:
                self.env_g2op.chronics_handler.reset()

        # Create the Gym environment
        self.env_gym = GymEnv(self.env_g2op)

        # Create custom observation and action spaces
        ## Redispatching agent
        self._aux_observation_spaces, self._aux_action_spaces = {}, {}
        if self.use_redispatching_agent:
            obs_space_kwargs, act_space_kwargs = get_normalization_kwargs(env.env_name)
            obs_attr_to_keep, act_attr_to_keep = obs_attr_to_keep_default.copy(), ["redispatch"]
            for op in ["add", "multiply"]: # Keep only redispatching in act_space_kwargs
                act_space_kwargs[op] = {"redispatch": act_space_kwargs[op]["redispatch"]}
            self._aux_observation_spaces.update({f"redispatching_agent": BoxGymnasiumObsSpace(env.observation_space,
                                                    attr_to_keep=obs_attr_to_keep,
                                                    **obs_space_kwargs
                                                )})
            self._aux_action_spaces.update({f"redispatching_agent": BoxGymnasiumActSpace(env.action_space,
                                                    attr_to_keep=act_attr_to_keep,
                                                    **act_space_kwargs
                                                )})
            # Normalizing observation and action spaces
            self._normalize_obs_or_act_space(self._aux_observation_spaces[f"redispatching_agent"], obs_space_kwargs, obs_attr_to_keep)
            self._normalize_obs_or_act_space(self._aux_action_spaces[f"redispatching_agent"], act_space_kwargs, act_attr_to_keep)

        ## Local agents
        for i in range(1, len(self.zone_names) + 1):
            obs_attr_to_keep, act_attr_to_keep, obs_space_kwargs, act_space_kwargs = get_obs_act_attr_and_kwargs(env, 
                                                                                                         self.zones_dict[self.zone_names[i-1]], 
                                                                                                         use_local_obs=not self.use_global_obs,
                                                                                                         add_local_redisp=self.add_local_redisp)
            self._aux_observation_spaces.update({f"agent_{i}": BoxGymnasiumObsSpace(env.observation_space,
                                                attr_to_keep=obs_attr_to_keep,
                                                **obs_space_kwargs
                                            )})
            self._aux_action_spaces.update({f"agent_{i}": BoxGymnasiumActSpace(env.action_space,
                                                attr_to_keep=act_attr_to_keep,
                                                **act_space_kwargs
                                            )})
            self._normalize_obs_or_act_space(self._aux_observation_spaces[f"agent_{i}"], obs_space_kwargs, obs_attr_to_keep)
            self._normalize_obs_or_act_space(self._aux_action_spaces[f"agent_{i}"], act_space_kwargs, act_attr_to_keep)

        # State space
        # Same as the redispatching agent observation space for the moment
        obs_space_kwargs, _ = get_normalization_kwargs(env.env_name)
        obs_attr_to_keep = obs_attr_to_keep_default.copy()
        self._aux_state_space = BoxGymnasiumObsSpace(env.observation_space,
                                                attr_to_keep=obs_attr_to_keep,
                                                **obs_space_kwargs)
        self._normalize_obs_or_act_space(self._aux_state_space, obs_space_kwargs, obs_attr_to_keep)
        
        # Initializing a state
        self._state = None

    def state(self):
        return self._state

    def _normalize_obs_or_act_space(self, space, space_kwargs, attr_to_keep):
         functs = space_kwargs.get("functs", {})
         divide = space_kwargs.get("divide", {})
         subtract = space_kwargs.get("subtract", {})
         add = space_kwargs.get("add", {})
         multiply = space_kwargs.get("multiply", {})
         for attr_nm in attr_to_keep:
                if ((attr_nm in divide) or (attr_nm in subtract)) or \
                   ((attr_nm in add) or (attr_nm in multiply)) or \
                    attr_nm in functs:
                    continue
                space.normalize_attr(attr_nm)
            

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent_id):
        return  Box(low=self._aux_observation_spaces[agent_id].low,
                           high=self._aux_observation_spaces[agent_id].high,
                           dtype=self._aux_observation_spaces[agent_id].dtype)
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent_id):
        return Box(low=self._aux_action_spaces[agent_id].low,
                           high=self._aux_action_spaces[agent_id].high,
                           dtype=self._aux_action_spaces[agent_id].dtype)

    def _map_agent_id_to_zone(self, agent_id):
        if agent_id == "redispatching_agent":
            return None
        else:
            idx = int(agent_id.split('_')[1]) - 1
            return self.zone_names[idx]

    def _update_env_g2op_config_for_rewards(self, env_g2op_config):
        if self._local_rewards is not None:
            self.use_global_reward = False
            # if "reward_kwargs" not in env_config:
            #     env_config["reward_kwargs"] = {agent_id:{} for agent_id in self.agents if agent_id != "redispatching_agent"}
            if self.use_redispatching_agent:
                env_g2op_config["reward_class"] = self._local_rewards["redispatching_agent"]
            env_g2op_config["other_rewards"] = {}
            for agent_id in self.agents:
                if agent_id != "redispatching_agent":
                    zone_agent = self._map_agent_id_to_zone(agent_id)
                    env_g2op_config["other_rewards"].update(
                        {agent_id: self._local_rewards[agent_id](zone_dict=self.zones_dict[zone_agent])})
        else:
            self.use_global_reward = True
            env_g2op_config.setdefault("reward_class", LinesCapacityReward)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._seed = seed
            self.np_random, seed = gym.utils.seeding.np_random(seed)
            self.env_g2op.seed(seed) 
            for agent_id in self.agents:
                self._aux_action_spaces[agent_id].seed(seed)
                self._aux_observation_spaces[agent_id].seed(seed)
                self.action_space(agent_id).seed(seed)
                self.observation_space(agent_id).seed(seed)
        # return observation dict and infos dict.
        # Cf gymenv dans gymcompat
        if (self._shuffle_chronics and 
                isinstance(self.env_g2op.chronics_handler.real_data, Multifolder) and 
                (not (options is not None and _TIME_SERIE_ID in options))):
                self.env_g2op.chronics_handler.sample_next_chronics()

        obs = self.env_g2op.reset()
        info = {agent_id: {} for agent_id in self.agents}
        gym_obs = self._to_gym_obs(obs)

        # Set state for the critic
        if self.use_redispatching_agent:
            self._state = gym_obs["redispatching_agent"]
        else:
            self._state = self._aux_state_space.to_gym(obs)

        return gym_obs, info

    def step(self, action_dict):
        # return observation dict, rewards dict, termination (done)/truncation (False) dicts, and infos dict
        g2op_act = self._from_gym_act(action_dict)
        obs_, rew_, done_, info = self.env_g2op.step(g2op_act)
        # Observations
        gym_obs = self._to_gym_obs(obs_)
        # Set state for the critic
        if self.use_redispatching_agent:
            self._state = gym_obs["redispatching_agent"]
        else:
            self._state = self._aux_state_space.to_gym(obs_)
        # Rewards
        if self.use_global_reward:
            rew = {agent_id: rew_ for agent_id in self.agents}
        else:
            rew = {agent_id: info["rewards"][agent_id] 
                   for agent_id in self.agents if agent_id != "redispatching_agent"}
            if self.use_redispatching_agent:
                rew.update({"redispatching_agent": rew_})
        # Termination
        done = {agent_id: done_ for agent_id in self.agents}
        truncated = {agent_id: False for agent_id in self.agents}
        info = {agent_id: {} for agent_id in self.agents}
        return gym_obs, rew, done, truncated, info
    
    def _to_gym_obs(self, grid2op_obs):
        # grid2op_obs is a common grid2op observation, it converts the grid2op observation to a gym one
        # return the proper dictionnary
        return {
            agent_id : self._aux_observation_spaces[agent_id].to_gym(grid2op_obs)
            for agent_id in self.agents
        }
    
    def _from_gym_act(self, gym_act_dict):
        # gym_act is a dictionnary of gym action, it converts the gym dict action to a grid2op one
        # return the proper grid2op action
        act = self.env_g2op.action_space({})  # empty action, to be filled later on
        for agent_id, gym_act in gym_act_dict.items():
            act += self._aux_action_spaces[agent_id].from_gym(gym_act)  # add the action of the agent to the global action
        return act
    
    def render(self):
        pass
    
    