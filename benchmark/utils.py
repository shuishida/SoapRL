from typing import Dict, Type

from stable_baselines3.common.base_class import BaseAlgorithm

from rl_zoo3.utils import ALGOS as rl_zoo3_algos
import sys


sys.path.insert(0, ".")
from ppoc.ppo import PPOC
from ppoem.ppo import PPOEM
from soap.ppo import SOAP
from dac.ppo import DAC


ALGOS: Dict[str, Type[BaseAlgorithm]] = {
    **rl_zoo3_algos,
    # Custom
    "ppoc": PPOC,
    "ppoem": PPOEM,
    "soap": SOAP,
    "dac": DAC,
}
