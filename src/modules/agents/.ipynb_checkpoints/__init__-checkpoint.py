REGISTRY = {}

from .rnn_agent import RNNAgent
from .rnn_ns_agent import RNNNSAgent
from .full_comm_agent import FullCommAgent
from .AC3T_agent import AC3TAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn_ns"] = RNNNSAgent
REGISTRY["full_comm"] = FullCommAgent
REGISTRY["AC3T"] = AC3TAgent