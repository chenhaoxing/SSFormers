from modules.query.relationnet import *
from modules.query.protonet import *
from modules.query.matchingnet import *
from modules.query.st import *
from modules.query.dn4 import *
import modules.registry as registry

def make_query(in_channels, cfg):
    return registry.Query[cfg.model.query](in_channels, cfg)
