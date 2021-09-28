from modules.semi_query.st import *
import modules.registry as registry

def make_query(in_channels, cfg):
    return registry.SemiQuery[cfg.model.query](in_channels, cfg)

