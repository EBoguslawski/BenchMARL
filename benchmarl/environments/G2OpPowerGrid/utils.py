import os
from typing import List, Dict, Any, Optional, Type, Tuple
import numpy as np
import json
import grid2op
from grid2op.dtypes import dt_int
from grid2op.utils import EpisodeStatistics


ENV_PATH = os.path.dirname(__file__)


with open(os.path.join(ENV_PATH, "zones_definitions.json"), "r", encoding="utf-8") as file:
    ZONES_DICT = json.load(file)


# attributes taken into account in the observation by default
obs_attr_to_keep_default = ["month", "day_of_week", "hour_of_day", "minute_of_hour",
                    "gen_p", "load_p", 
                    "p_or", "rho", "timestep_overflow", "line_status",
                    # dispatch part of the observation
                    "actual_dispatch", "target_dispatch",
                    # storage part of the observation
                    "storage_charge", "storage_power",
                    # curtailment part of the observation
                    "curtailment", "curtailment_limit",  "gen_p_before_curtail",
                    ]
# attributes of the possible actions by default
act_attr_to_keep_default = ["curtail", "set_storage"]


def add_missing_keys(zones_dict: Dict[str, Any]):
    zone_names = zones_dict.keys()
    every_keys = {key for zone_name in zone_names for key in zones_dict[zone_name].keys()}
    missing_keys_zones = [every_keys - set(zones_dict[zone_name].keys()) for zone_name in zone_names]
    for zone_name, missing_keys in zip(zone_names, missing_keys_zones):
        for missing_key in missing_keys:
            zones_dict[zone_name][missing_keys] = []
    return zones_dict

def get_zone_dict_from_path(zones_path: Optional[str] = None) -> Dict[str, Any]:
    if zones_path is not None:
            with open(zones_path, "r", encoding="utf-8") as file:
                zones_dict = json.load(file)
    else:
        zones_dict = ZONES_DICT
    return zones_dict

def get_zone_dict_from_names(zone_names: List[str], zones_path: Optional[str] = None) -> Dict[str, Any]:

    if isinstance(zone_names, str):
        zone_names = [zone_names]

    if "Complete" in zone_names: 
        if len(zone_names) == 1:
            return None
        else:
            raise ValueError(f"If 'Complete' is in zone_names, then it has to be the only zone. You entered {len(zone_names)} zones.")
        
    zones_dict = get_zone_dict_from_path(zones_path=zones_path)

    for zone_name in zone_names:
        if zone_name not in zones_dict.keys():
            raise ValueError(f"The zone name {zone_name} is unkown. Possible zone names are {list(zones_dict.keys())}.")

    if len(zone_names) == 1:
        final_dict = zones_dict[zone_names[0]]
    else:
    # Add missing keys to each zone dictionnary
        every_keys = {key for zone_name in zone_names for key in zones_dict[zone_name].keys()}
        missing_keys_zones = [every_keys - set(zones_dict[zone_name].keys()) for zone_name in zone_names]
        for zone_name, missing_keys in zip(zone_names, missing_keys_zones):
            for missing_key in missing_keys:
                zones_dict[zone_name][missing_keys] = []
        final_dict = {}
        for key in every_keys:
            val_tmp = [zones_dict[zone_name][key] for zone_name in zone_names]
            val_tmp = np.concatenate(val_tmp)
            val_tmp = np.unique(val_tmp)
            final_dict[key] = [int(el) for el in np.sort(val_tmp)] # converting back to an int list because np.arrays are not jsonable and we need int type numbers

    return final_dict


def normalize_attr_obs(vect, attr_name: str, dict: Dict[str, Any], idx: int):
    if attr_name in dict["subtract"].keys():
        substract = np.array(dict["subtract"][attr_name])
        divide = np.array(dict["divide"][attr_name])
        res = (vect[idx] - substract[idx])/divide[idx]
    else:
        res = vect
    return res


def get_normalization_kwargs(env_name: str) -> Dict[str, Any]:
    if "l2rpn_idf_2023" in env_name:
        path = os.path.join(ENV_PATH, "normalization/l2rpn_idf_2023")
    elif "l2rpn_wcci_2022" in env_name:
        path = os.path.join(ENV_PATH, "normalization/l2rpn_wcci_2022")
    else:
        raise ValueError(f"""The normalization kwargs are not precomputed for this environment. 
                         You entered an environement named {env_name}, but this function only
                         handles environment derived from l2rpn_idf_2023 and l2rpn_wcci_2022.""")

    with open(os.path.join(path, "preprocess_obs.json"), "r", encoding="utf-8") as f:
        obs_space_kwargs = json.load(f)
    with open(os.path.join(path, "preprocess_act.json"), "r", encoding="utf-8") as f:
        act_space_kwargs = json.load(f)

    return obs_space_kwargs, act_space_kwargs


def get_obs_act_attr_and_kwargs(env, zone_dict: Dict[str, Any], add_storage_setpoint: bool = False, use_local_obs: bool = True) -> Tuple[Dict[str, Any], ...]:
    
    obs_space_kwargs, _ = get_normalization_kwargs(env.env_name)
    
    line_large_idx = zone_dict["line_large_idx"]
        
    # substations_inside_idx = [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 65, 66]
    # substations_border_idx = [48, 64]

    gen_inside_idx = zone_dict["gen_inside_idx"]
    gen_large_idx = zone_dict["gen_large_idx"]
    gen_curtail_inside_idx = np.intersect1d(gen_inside_idx, np.where(env.gen_renewable))
    gen_redisp_inside_idx = np.intersect1d(gen_inside_idx, np.where(env.gen_redispatchable))
    gen_curtail_large_idx = np.intersect1d(gen_large_idx, np.where(env.gen_renewable))
    gen_redisp_large_idx = np.intersect1d(gen_large_idx, np.where(env.gen_redispatchable))

    load_large_idx = zone_dict["load_large_idx"]

    storage_inside_idx = zone_dict["storage_inside_idx"]
    storage_border_idx = zone_dict["storage_border_idx"]

    
    act_attr_to_keep = [
                        "curtail_zone", 
                        "set_storage_zone",
                        ]
    
    n_storage = env.n_storage # we must not use the env directly in the lambda function otherwise the model becomes very heavy when saving
    if not use_local_obs:
        obs_attr_to_keep = ["month", "day_of_week", "hour_of_day", "minute_of_hour",
                        "gen_p", "load_p", 
                        "p_or", "rho", "timestep_overflow", "line_status",
                        # dispatch part of the observation
                        "actual_dispatch", "target_dispatch",
                        # storage part of the observation
                        "storage_charge", "storage_power",
                        # curtailment part of the observation
                        "curtailment", "curtailment_limit",  "gen_p_before_curtail",
                        ]

        obs_space_kwargs_zone = obs_space_kwargs.copy()
    else:
        obs_attr_to_keep = ["month", "day_of_week", "hour_of_day", "minute_of_hour",
                        "gen_p_zone", "load_p_zone", 
                        "p_or_zone", "rho_zone", "timestep_overflow_zone", "line_status_zone",
                        # # dispatch part of the observation
                        "actual_dispatch_zone", "target_dispatch_zone",
                        # # storage part of the observation
                        # for the storage units, we separate the indexes inside and at the border of the zone because it makes easier the implementation of the setpoint feature
                        "storage_charge_zone", "storage_power_zone", 
                        "storage_charge_zone_border", "storage_power_zone_border",
                        # # curtailment part of the observation
                        "curtailment_zone", "curtailment_limit_zone", "gen_p_before_curtail_zone",
                        ]
        obs_space_kwargs_zone = {}
        obs_space_kwargs_zone["functs"] ={
                                "gen_p_zone": (lambda grid2op_obs: normalize_attr_obs(grid2op_obs.gen_p, "gen_p", obs_space_kwargs, gen_large_idx),
                                                            None, None, None, None),
                                 "load_p_zone": (lambda grid2op_obs: normalize_attr_obs(grid2op_obs.load_p, "load_p", obs_space_kwargs, load_large_idx),
                                                            None, None, None, None),                                 
                                 "p_or_zone": (lambda grid2op_obs: normalize_attr_obs(grid2op_obs.p_or, "p_or", obs_space_kwargs, line_large_idx),
                                                            None, None, None, None), 
                                 "rho_zone": (lambda grid2op_obs: normalize_attr_obs(grid2op_obs.rho, "rho", obs_space_kwargs, line_large_idx),
                                                            None, None, None, None), 
                                 "timestep_overflow_zone": (lambda grid2op_obs: grid2op_obs.timestep_overflow[line_large_idx],
                                                            np.full(shape=(len(line_large_idx),), fill_value=0, dtype=dt_int), 
                                                            np.full(shape=(len(line_large_idx),), fill_value=np.iinfo(dt_int).max, dtype=dt_int), 
                                                            None, dt_int), # TODO Handle integer type
                                 "line_status_zone": (lambda grid2op_obs: grid2op_obs.line_status[line_large_idx],
                                                            np.full(shape=(len(line_large_idx),), fill_value=0, dtype=dt_int), 
                                                            np.full(shape=(len(line_large_idx),), fill_value=1, dtype=dt_int), None, dt_int),
                                # # dispatch part of the observation
                                "actual_dispatch_zone": (lambda grid2op_obs: grid2op_obs.actual_dispatch[gen_redisp_large_idx] / np.maximum(-grid2op_obs.gen_pmin, grid2op_obs.gen_pmax)[gen_redisp_large_idx],
                                                            None, None, None, None),
                                "target_dispatch_zone": (lambda grid2op_obs: grid2op_obs.gen_p[gen_redisp_large_idx] / np.maximum(-grid2op_obs.gen_pmin, grid2op_obs.gen_pmax)[gen_redisp_large_idx],
                                                            None, None, None, None),
                                # storage part of the observation
                                "storage_charge_zone": (lambda grid2op_obs: grid2op_obs.storage_charge[storage_inside_idx] / grid2op_obs.storage_Emax[storage_inside_idx],
                                                            None, None, None, None), 
                                "storage_power_zone": (lambda grid2op_obs: grid2op_obs.storage_power[storage_inside_idx] / grid2op_obs.storage_max_p_prod[storage_inside_idx],
                                                            None, None, None, None),
                                "storage_charge_zone_border": (lambda grid2op_obs: grid2op_obs.storage_charge[storage_border_idx] / grid2op_obs.storage_Emax[storage_border_idx],
                                                            None, None, None, None), 
                                "storage_power_zone_border": (lambda grid2op_obs: grid2op_obs.storage_power[storage_border_idx] / grid2op_obs.storage_max_p_prod[storage_border_idx],
                                                            None, None, None, None),
                                # # curtailment part of the observation
                                "curtailment_zone": (lambda grid2op_obs: grid2op_obs.curtailment[gen_curtail_large_idx],
                                                            None, None, None, None),
                                "curtailment_limit_zone": (lambda grid2op_obs: grid2op_obs.curtailment_limit[gen_curtail_large_idx],
                                                            None, None, None, None),
                                "gen_p_before_curtail_zone": (lambda grid2op_obs: normalize_attr_obs(grid2op_obs.gen_p_before_curtail, "gen_p_before_curtail", obs_space_kwargs, gen_large_idx),
                                                            None, None, None, None),                                 

                                 }
    
    if add_storage_setpoint:
        # Add to observation space the "storage_setpoint" attribute, the setpoint we have to follow
        obs_attr_to_keep.append("storage_setpoint")
        obs_space_kwargs_zone["functs"]["storage_setpoint"] = (lambda grid2op_obs: np.zeros(len(storage_inside_idx)),
                                                            0., 1., None, None)

        # Add a needed argument in the gym observation space with setpoint
        storages_of_interest = np.full(n_storage, False)
        storages_of_interest[storage_inside_idx] = True
        obs_space_kwargs_zone["storages_of_interest"] = storages_of_interest
    
    grid2op_act_do_nothing = env.action_space({}) # we must not use the env directly in the lambda function otherwise the model becomes very heavy when saving
    # we use a grid2op action to create ours instead
    def from_gym_act_curtail(gym_act_curtail):
        curtail_vect = [(gen_id, curt) for gen_id, curt in zip(gen_curtail_inside_idx, gym_act_curtail)]
        grid2op_act_curtail = grid2op_act_do_nothing.copy()
        grid2op_act_curtail.curtail = curtail_vect
        return grid2op_act_curtail
    
    storage_max_p_prod = env.storage_max_p_prod # we must not use the env directly in the lambda function otherwise the model becomes very heavy when saving
    def from_gym_act_storage(gym_act_storage):
        storage_vect = np.zeros(n_storage)
        storage_vect[storage_inside_idx] = gym_act_storage * storage_max_p_prod[storage_inside_idx]
        grid2op_act_storage = grid2op_act_do_nothing.copy()
        grid2op_act_storage.storage_p = storage_vect
        return grid2op_act_storage
    
    act_space_kwargs_zone = {}
    act_space_kwargs_zone["functs"] ={
                                "curtail_zone": (from_gym_act_curtail, 0., 1., (len(gen_curtail_inside_idx),), None),
                                 "set_storage_zone": (from_gym_act_storage, -1., 1., (len(storage_inside_idx),), None),
                                 
            }
    
    return obs_attr_to_keep, act_attr_to_keep, obs_space_kwargs_zone, act_space_kwargs_zone


def _aux_get_stat_metadata(env_name: str, dn: bool = True, name_stat: Optional[str] = None) -> Dict[str, Any]:
    if os.path.isdir(env_name):
        path_env = env_name
    else:
        path_ = grid2op.get_current_local_dir()
        path_env = os.path.join(path_, env_name)
    if not os.path.exists(path_env):
        raise RuntimeError(f"The environment \"{env_name}\" does not exist.")

    path_dn = os.path.join(path_env, "_statistics_l2rpn_dn")
        
    if not os.path.exists(path_dn):
        raise RuntimeError("The folder _statistics_icaps2021_dn (or _statistics_l2rpn_dn) used for computing the score do not exist")
    path_reco = os.path.join(path_env, "_statistics_l2rpn_no_overflow_reco")
    if not os.path.exists(path_reco):
        raise RuntimeError("The folder _statistics_l2rpn_no_overflow_reco used for computing the score do not exist")
    
    if name_stat is None:
        if dn:
            path_metadata = os.path.join(path_dn, "metadata.json")
        else:
            path_metadata = os.path.join(path_reco, "metadata.json")
    else:
        path_stat = os.path.join(path_env, EpisodeStatistics.get_name_dir(name_stat))
        if not os.path.exists(path_stat):
            raise RuntimeError(f"No folder associated with statistics {name_stat}")
        path_metadata = os.path.join(path_stat, "metadata.json")
    
    if not os.path.exists(path_metadata):
        raise RuntimeError("No metadata can be found for the statistics you wanted to compute.")
    
    with open(path_metadata, "r", encoding="utf-8") as f:
        dict_ = json.load(f)
    
    return dict_


def get_env_seed(env_name: str) -> List[int]:
    """This function ensures that you can reproduce the results of the computed scenarios.
    
    It forces the seeds of the environment, during evaluation to be the same as the one used during the evaluation of the score.
    
    As environments are stochastic in grid2op, it is very important that you use this function (or a similar one) before
    computing the scores of your agent.

    Args:
        env_name (str): The environment name on which you want to retrieve the seeds used

    Raises:
        RuntimeError: When it is not possible to retrieve the seeds (for example when the "statistics" has not been computed)

    Returns:
        List[int]: List of environment's seeds saved in the statistics folder during the score initialization
    """

    dict_ = _aux_get_stat_metadata(env_name)
    
    key = "env_seeds"
    if key not in dict_:
        raise RuntimeError(f"Impossible to find the key {key} in the metadata dictionnary. You should re run the score function.")
    
    return dict_[key]
    