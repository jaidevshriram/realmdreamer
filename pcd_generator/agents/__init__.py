from .dummy import DummyAgent


def setup_agent(cfg):
    if cfg.name == "dummy":
        return DummyAgent(cfg)
    else:
        raise NotImplementedError(f"Agent type {cfg.name} not implemented")
