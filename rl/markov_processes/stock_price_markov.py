from dataclasses import dataclass
from typing import Optional, Mapping
from rl.gen_utils.common_funcs import get_logistic_func, get_unit_sigmoid_func
from rl.distribution import Categorical, Constant
from rl.markov_processes.stock_price_simulation import (
    plot_single_trace_all_processes,
    plot_distribution_at_time_all_processes,
)
from rl.markov_process import State, NonTerminal, MarkovProcess

import numpy as np
import itertools


@dataclass(frozen=True)
class StateMP1:
    """
    For our first process, we assume our state to be just the price.
    Hence S[t] = (X[t])
    where X[t] is the Price at time t.
    """

    price: int


@dataclass
class StockPriceMP1(MarkovProcess[StateMP1]):
    """
    The assumption for the first process is that the price tends to revert to
    a mean that is defined.

    Level is the average level that the price tends to "mean revert" to.
    Alpha1 is the "strength" or magnitude of this mean-reverting behaviour.
    """

    level_param: int
    alpha1: float = 0.25

    def up_prob(self, state: StateMP1) -> float:
        """
        The model that we implement for the first process is
        P(S[t+1] | S[t]) = 1.0 / (1 + e ^ ((-alpha1) * (level - X[t])))
        """
        return get_logistic_func(self.alpha1)(self.level_param - state.price)

    def transition(self, state: NonTerminal[StateMP1]) -> Categorical[State[StateMP1]]:
        """
        The transition probabilities for the states given the probability that the price goes up
        is generated for both Price[t] + 1 as up_probability and Price[t] - 1 as 1 - up_probability.
        """
        up_p = self.up_prob(state.state)
        return Categorical(
            {
                NonTerminal(StateMP1(state.state.price + 1)): up_p,
                NonTerminal(StateMP1(state.state.price - 1)): 1 - up_p,
            }
        )


handy_map: Mapping[Optional[bool], int] = {True: -1, False: 1, None: 0}


@dataclass(frozen=True)
class StateMP2:
    """
    For the second process, we utilize 2 state variables - Price[t] and a boolean is_prev_move_up,
    which represents if the last price movement was up or down.
    Hence S[t] = (X[t], X[t] - X[t-1])
    where X[t] is the Price at time t.
    """

    price: int
    is_prev_move_up: Optional[bool]


@dataclass
class StockPriceMP2(MarkovProcess[StateMP2]):
    """
    The model we implement for the second process assumes that the price will move in reverse bias
    to the last price.
    Alpha2 is magnitude by which this reverse bias occurs.
    """

    alpha2: float = 0.75

    def up_prob(self, state: StateMP2) -> float:
        """
        The model that we implement for the second process is
        P(S[t+1] | S[t]) = 0.5 * (1 + alpha2 * (direction of previous move)) if t > 0
        and P(S[t+1] | S[t]) = 0.5 if t = 0
        """
        return 0.5 * (1 + self.alpha2 * handy_map[state.is_prev_move_up])

    def transition(self, state: NonTerminal[StateMP2]) -> Categorical[State[StateMP2]]:
        """
        The transition probabilities for the states given the probability that the price goes up
        is generated for both Price[t] + 1 as up_probability and Price[t] - 1 as 1 - up_probability.
        """
        up_p = self.up_prob(state.state)
        return Categorical(
            {
                NonTerminal(
                    StateMP2(state.state.price + 1, state.state.is_prev_move_up)
                ): up_p,
                NonTerminal(
                    StateMP2(state.state.price - 1, state.state.is_prev_move_up)
                ): 1
                - up_p,
            }
        )


@dataclass(frozen=True)
class StateMP3:
    """
    For the third process, we utilize 2 state variables - num_up_moves and num_down_moves.
    These 2 variables represent the number of up_moves and down_moves that the price has taken
    from the beginning.
    U[t] = Sum[i=1,t](max(X[i] - X[i - 1], 0))
    D[t] = Sum[i=1,t](max(X[i - 1] - X[i], 0))
    and,
    S[t] = (U[t], D[t])
    where,
    U[t] is the number of up moves.
    D[t] is the number of down moves.
    """

    num_up_moves: int
    num_down_moves: int


@dataclass
class StockPriceMP3(MarkovProcess[StateMP3]):
    """
    The model we implement for the third process assumes that the price will depend on
    all past movements. Specifically, it depends on number of past up moves relative to the
    number of past down moves.
    Alpha3 is magnitude by which this bias occurs.
    """

    alpha3: float = 1.0

    def up_prob(self, state: StateMP3) -> float:
        """
        The model that we implement for the third process is
        P(S[t+1] | S[t]) =  1 / ( 1 + ( ( U[t] + D[t] ) / D[t] - 1 ) ^ alpha3 ) if t > 0
        P(S[t+1] | S[t]) = 0.5 if t = 0
        """
        total = state.num_down_moves + state.num_up_moves

        return (
            get_unit_sigmoid_func(self.alpha3)(state.num_down_moves / total)
            if total
            else 0.5
        )

    def transition(self, state: NonTerminal[StateMP3]) -> Categorical[State[StateMP3]]:
        """
        The transition probabilities for the states given the probability that the price goes up
        is generated for both Price[t] + 1 as up_probability and Price[t] - 1 as 1 - up_probability.
        """
        up_p = self.up_prob(state.state)
        return Categorical(
            {
                NonTerminal(
                    StateMP3(state.state.num_up_moves + 1, state.state.num_down_moves)
                ): up_p,
                NonTerminal(
                    StateMP3(state.state.num_up_moves, state.state.num_down_moves + 1)
                ): 1
                - up_p,
            }
        )


def process1_price_traces(
    start_price: int, level_param: int, alpha1: float, time_steps: int, num_traces: int
) -> np.ndarray:
    """
    This function generates the price traces for Process 1.
    """
    mp = StockPriceMP1(level_param, alpha1)
    # Instantiate a Distribution with a single outcome of probability 1.
    start_state_distribution = Constant(NonTerminal(StateMP1(start_price)))

    # Generate the number of traces as passed as parameter.
    return np.vstack(
        [
            # Iterate over all simulated price changes given a single state
            # from t = 1 to time steps.
            np.fromiter(
                (
                    s.state.price
                    for s in itertools.islice(
                        # Simulate the markov process as given by the transition function.
                        mp.simulate(start_state_distribution),
                        time_steps + 1,
                    )
                ),
                float,
            )
            for _ in range(num_traces)
        ]
    )


def process2_price_traces(
    start_price: int, alpha2: float, time_steps: int, num_traces: int
) -> np.ndarray:
    """
    This function generates the price traces for Process 2.
    """
    mp = StockPriceMP2(alpha2)
    # Instantiate a Distribution with a single outcome of probability 1.
    start_state_distribution = Constant(NonTerminal(StateMP2(start_price, None)))

    # Generate the number of traces as passed as parameter.
    return np.vstack(
        [
            # Iterate over all simulated price changes given a single state
            # from t = 1 to time steps.
            np.fromiter(
                (
                    s.state.price
                    for s in itertools.islice(
                        # Simulate the markov process as given by the transition function.
                        mp.simulate(start_state_distribution),
                        time_steps + 1,
                    )
                ),
                float,
            )
            for _ in range(num_traces)
        ]
    )


def process3_price_traces(
    start_price: int, alpha3: float, time_steps: int, num_traces: int
) -> np.ndarray:
    """
    This function generates the price traces for Process 2.
    """
    mp = StockPriceMP3(alpha3)
    # Instantiate a Distribution with a single outcome of probability 1.
    start_state_distribution = Constant(
        NonTerminal(StateMP3(num_up_moves=0, num_down_moves=0))
    )

    # Generate the number of traces as passed as parameter.
    return np.vstack(
        [
            # Iterate over all simulated price changes given a single state
            # from t = 1 to time steps.
            np.fromiter(
                (
                    start_price + s.state.num_up_moves - s.state.num_down_moves
                    for s in itertools.islice(
                        # Simulate the markov process as given by the transition function.
                        mp.simulate(start_state_distribution),
                        time_steps + 1,
                    )
                ),
                float,
            )
            for _ in range(num_traces)
        ]
    )


if __name__ == "__main__":
    start_price: int = 100
    level_param: int = 100
    alpha1: float = 0.25
    alpha2: float = 0.75
    alpha3: float = 1.0
    time_steps: int = 100
    num_traces: int = 1000

    process1_traces: np.ndarray = process1_price_traces(
        start_price=start_price,
        level_param=level_param,
        alpha1=alpha1,
        time_steps=time_steps,
        num_traces=num_traces,
    )
    process2_traces: np.ndarray = process2_price_traces(
        start_price=start_price,
        alpha2=alpha2,
        time_steps=time_steps,
        num_traces=num_traces,
    )
    process3_traces: np.ndarray = process3_price_traces(
        start_price=start_price,
        alpha3=alpha3,
        time_steps=time_steps,
        num_traces=num_traces,
    )

    trace1 = process1_traces[0]
    trace2 = process2_traces[0]
    trace3 = process3_traces[0]

    plot_single_trace_all_processes(trace1, trace2, trace3)

    plot_distribution_at_time_all_processes(
        process1_traces, process2_traces, process3_traces
    )
