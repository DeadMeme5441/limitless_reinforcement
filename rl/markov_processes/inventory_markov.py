from dataclasses import dataclass
from typing import Dict, Mapping
from rl.distribution import Categorical, FiniteDistribution
from scipy.stats import poisson

from rl.markov_process import FiniteMarkovProcess


@dataclass(frozen=True)
class InventoryState:
    """
    This is the State the markov process has access to.
    It consists of on_hand stock and on_order stock.
    """

    on_hand: int
    on_order: int

    def inventory_position(self) -> int:
        """
        This is the total inventory position that is the
        sum of the on_hand stock and on_order stock.
        """
        return self.on_hand + self.on_order


class InventoryMPFinite(FiniteMarkovProcess[InventoryState]):
    def __init__(self, capacity: int, poisson_lambda: float):
        self.capacity: int = capacity
        self.poisson_lambda: float = poisson_lambda

        self.poisson_distr = poisson(poisson_lambda)
        super().__init__(self.get_transition_map())

    def get_transition_map(
        self,
    ) -> Mapping[InventoryState, FiniteDistribution[InventoryState]]:
        d: Dict[InventoryState, Categorical[InventoryState]] = {}
        for alpha in range(self.capacity + 1):
            for beta in range(self.capacity + 1 - alpha):
                state = InventoryState(alpha, beta)
                ip = state.inventory_position()
                beta1 = self.capacity - ip
                state_probs_map: Mapping[InventoryState, float] = {
                    InventoryState(ip - i, beta1): (
                        self.poisson_distr.pmf(i)
                        if i < ip
                        else 1 - self.poisson_distr.cdf(ip - 1)
                    )
                    for i in range(ip + 1)
                }
                d[InventoryState(alpha, beta)] = Categorical(state_probs_map)

        return d


if __name__ == "__main__":
    user_capacity = 2
    user_poisson_lambda = 1.0

    si_mp = InventoryMPFinite(
        capacity=user_capacity, poisson_lambda=user_poisson_lambda
    )

    print("Transition Map")
    print("--------------")
    print(si_mp)

    print("Stationary Distribution")
    print("-----------------------")
    si_mp.display_stationary_distribution()
