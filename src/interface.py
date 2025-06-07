from .ADASYNBag import ADASYNBag
from .BBag import BBag
from .BEBS import BEBS
from .BEV import BEV
from .CostBag import CostBag
from .EBBag import EBBag
from .EUSBag import EUSBag
from .LazyBag import LazyBag
from .MRBBag import MRBBag
from .MultiRandBalBag import MultiRandBalBag
from .NBBag import NBBag
from .OverBag import OverBag
from .RSYNBag import RSYNBag
from .PTBag import PTBag
from .RBBag import RBBag
from .RBSBag import RBSBag
from .REABag import REABag
from .SMOTEBag import SMOTEBag
from .UnderBag import UnderBag
from .UnderBagKNN import UnderBagKNN

Imbalance_BAGGING_ALGORITHMS = {
    'ADASYNBag': ADASYNBag,
    'BBag' : BBag,
    'BEBS' : BEBS,
    'BEV' : BEV,
    'CostBag' : CostBag,
    'EBBag' : EBBag,
    'EUSBag' : EUSBag,
    'LazyBag' : LazyBag,
    'MRBBag' : MRBBag,
    'MultiRandBalBag' : MultiRandBalBag,
    'NBBag' : NBBag,
    'OverBag' : OverBag,
    'RSYNBag' : RSYNBag,
    'PTBag' : PTBag,
    'RBBag' : RBBag,
    'RBBag' : RBBag,
    'RBSBag' : RBSBag,
    'REABag' : REABag,
    'SMOTEBag' : SMOTEBag,
    'UnderBag' : UnderBag,
    'UnderBagKNN' :  UnderBagKNN
}


def get_imbalance_bagging_classifier(algorithm='BBag', **kwargs):
    if algorithm.lower() == 'bbag':
        return BBag(**kwargs)
    elif algorithm.lower() == 'adasynbag':
        return ADASYNBag(**kwargs)
    elif algorithm.lower() == 'bbag':
        return BBag(**kwargs)
    elif algorithm.lower() == 'bebs':
        return BEBS(**kwargs)
    elif algorithm.lower() == 'bev':
        return BEV(**kwargs)
    elif algorithm.lower() == 'costbag':
        return CostBag(**kwargs)
    elif algorithm.lower() == 'ebbag':
        return EBBag(**kwargs)
    elif algorithm.lower() == 'eusbag':
        return EUSBag(**kwargs)
    elif algorithm.lower() == 'lazybag':
        return LazyBag(**kwargs)
    elif algorithm.lower() == 'mrbbag':
        return MRBBag(**kwargs)
    elif algorithm.lower() == 'multirandbalbag':
        return MultiRandBalBag(**kwargs)
    elif algorithm.lower() == 'nbbag':
        return NBBag(**kwargs)
    elif algorithm.lower() == 'overbag':
        return OverBag(**kwargs)
    elif algorithm.lower() == 'ptbag':
        return PTBag(**kwargs)
    elif algorithm.lower() == 'rbbag':
        return RBBag(**kwargs)
    elif algorithm.lower() == 'rbsbag':
        return RBSBag(**kwargs)
    elif algorithm.lower() == 'reabag':
        return REABag(**kwargs)
    elif algorithm.lower() == 'rsynbag':
        return RSYNBag(**kwargs)
    elif algorithm.lower() == 'smotebag':
        return SMOTEBag(**kwargs)
    elif algorithm.lower() == 'underbag':
        return UnderBag(**kwargs)
    elif algorithm.lower() == 'underbagknn':
        return UnderBagKNN(**kwargs)
    else:
        raise ValueError(f"Algorithm {algorithm} is not recognized or implemented.")