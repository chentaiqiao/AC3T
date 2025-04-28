from .q_learner import QLearner
from .qtran_learner import QLearner as QTranLearner
from .actor_critic_learner import ActorCriticLearner
from .dmaq_qatten_learner import DMAQ_qattenLearner
from .AC3T_learner import AC3TLearner
from .AC3T_qplex_learner import AC3TLearner as AC3TQPLEXLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["actor_critic_learner"] = ActorCriticLearner
REGISTRY["dmaq_qatten_learner"] = DMAQ_qattenLearner
REGISTRY["AC3T_learner"] = AC3TLearner
REGISTRY["AC3T_qplex_learner"] = AC3TQPLEXLearner