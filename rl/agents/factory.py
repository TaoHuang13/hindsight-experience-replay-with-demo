from rl.agents.herdemo import HerDemo

AGENTS = {
    'HerDemo': HerDemo,
}

def make_agent(env_params, sampler, cfg):
    cfg.obs_dim_o = env_params['obs']
    cfg.obs_dim_g = env_params['goal']
    cfg.act_dim = env_params['act']

    if cfg.name not in AGENTS.keys():
        assert 'agent is not supported: %s' % cfg.name
    else:
        return AGENTS[cfg.name](
            env_param=env_params,
            sampler=sampler,
            agent_cfg=cfg
        )