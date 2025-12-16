from dataclasses import dataclass, MISSING


@dataclass
class TaskConfig:
    # map_size: int = MISSING
    # minimap_mode: bool = MISSING
    # tag_penalty: float = MISSING
    # max_cycles: int = MISSING
    # extra_features: bool = MISSING
    env_name: str = MISSING
    zone_names: list = MISSING
    use_global_obs: bool = MISSING
    use_redispatching_agent: bool = MISSING
    add_local_redisp: bool = MISSING
    env_g2op_config: dict = MISSING
    local_rewards: dict = MISSING
    shuffle_chronics: bool = MISSING
    regex_filter_chronics: str = MISSING
    zones_path: str = MISSING
    safe_max_rho: float = MISSING
    curtail_margin: float = MISSING
