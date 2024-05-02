import numpy as np
from dataclasses import dataclass
from typing import Optional, Mapping, Sequence, Tuple
import itertools


@dataclass
class Process1:
    @dataclass
    class State:
        price: int

    level_param: int
    alpha1: float = 0.25

    def up_prob(self, state: State) -> float:
        return 1.0 / (1 + np.exp(-self.alpha1 * (self.level_param - state.price)))

    def next_state(self, state: State) -> State:
        up_move: int = np.random.binomial(1, self.up_prob(state), 1)[0]
        return Process1.State(price=state.price + up_move * 2 - 1)


handy_map: Mapping[Optional[bool], int] = {True: -1, False: 1, None: 0}


@dataclass
class Process2:
    @dataclass
    class State:
        price: int
        is_prev_move_up: Optional[bool]

    alpha2: float = 0.75

    def up_prob(self, state: State) -> float:
        return 0.5 * (1 + self.alpha2 * handy_map[state.is_prev_move_up])

    def next_state(self, state: State) -> State:
        up_move: int = np.random.binomial(1, self.up_prob(state), 1)[0]
        return Process2.State(
            price=state.price + up_move * 2 - 1, is_prev_move_up=bool(up_move)
        )


@dataclass
class Process3:
    @dataclass
    class State:
        num_up_moves: int
        num_down_moves: int

    alpha3: float = 1.0

    def up_prob(self, state: State) -> float:
        total = state.num_up_moves + state.num_down_moves
        if total == 0:
            return 0.5
        elif state.num_down_moves == 0:
            return state.num_down_moves**self.alpha3
        else:
            return 1.0 / (1 + (total / state.num_down_moves - 1) ** self.alpha3)

    def next_state(self, state: State) -> State:
        up_move: int = np.random.binomial(1, self.up_prob(state), 1)[0]
        return Process3.State(
            num_up_moves=state.num_up_moves + up_move,
            num_down_moves=state.num_down_moves + 1 - up_move,
        )


def simulation(process, start_state):
    state = start_state
    while True:
        yield state
        state = process.next_state(state)


def process1_price_traces(
    start_price: int, level_param: int, alpha1: float, time_steps: int, num_traces: int
) -> np.ndarray:
    process = Process1(level_param, alpha1)

    start_state = Process1.State(price=start_price)
    return np.vstack(
        [
            np.fromiter(
                (
                    s.price
                    for s in itertools.islice(
                        simulation(process, start_state), time_steps + 1
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
    process = Process2(alpha2)

    start_state = Process2.State(price=start_price, is_prev_move_up=None)
    return np.vstack(
        [
            np.fromiter(
                (
                    s.price
                    for s in itertools.islice(
                        simulation(process, start_state), time_steps + 1
                    )
                ),
                float,
            )
            for _ in range(num_traces)
        ]
    )


def process3_price_traces(
    start_price: int, alpha3: float, time_steps: int, num_traces: int
):
    process = Process3(alpha3)

    start_state = Process3.State(num_up_moves=0, num_down_moves=0)

    return np.vstack(
        [
            np.fromiter(
                (
                    start_price + s.num_up_moves - s.num_down_moves
                    for s in itertools.islice(
                        simulation(process, start_state), time_steps + 1
                    )
                ),
                float,
            )
            for _ in range(num_traces)
        ]
    )


if __name__ == "__main__":
    trace1 = process1_price_traces(100, 100, 0.25, 100, 1000)[0]
    trace2 = process2_price_traces(100, 0.25, 100, 1000)[0]
    trace3 = process3_price_traces(100, 1.0, 100, 1000)[0]

    print(trace3)
