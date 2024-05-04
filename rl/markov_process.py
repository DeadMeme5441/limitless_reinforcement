from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Callable, Iterable, Sequence, TypeVar, Mapping, Set
from rl.distribution import Categorical, Distribution, FiniteDistribution
import numpy as np
from pprint import pprint
import graphviz

S = TypeVar("S")
X = TypeVar("X")


class State(ABC, Generic[S]):
    state: S

    def on_non_terminal(self, f: Callable[["NonTerminal[S]"], X], default: X) -> X:

        if isinstance(self, NonTerminal):
            return f(self)
        else:
            return default


@dataclass(frozen=True)
class Terminal(State[S]):
    state: S


@dataclass(frozen=True)
class NonTerminal(State[S]):
    state: S

    def __eq__(self, other):
        return self.state == other.state

    def __lt__(self, other):
        return self.state < other.state


class MarkovProcess(ABC, Generic[S]):
    @abstractmethod
    def transition(self, state: NonTerminal[S]) -> Distribution[State[S]]:
        pass

    def simulate(
        self, start_state_distribution: Distribution[NonTerminal[S]]
    ) -> Iterable[State[S]]:
        state: State[S] = start_state_distribution.sample()
        yield state

        while isinstance(state, NonTerminal):
            state = self.transition(state).sample()
            yield state


Transition = Mapping[NonTerminal[S], FiniteDistribution[State[S]]]


class FiniteMarkovProcess(MarkovProcess[S]):
    non_terminal_states: Sequence[NonTerminal[S]]
    transition_map: Transition[S]

    def __init__(self, transition_map: Mapping[S, FiniteDistribution[S]]):
        non_terminals: Set[S] = set(transition_map.keys())
        self.transition_map = {
            NonTerminal(s): Categorical(
                {
                    (NonTerminal(s1) if s1 in non_terminals else Terminal(s1)): p
                    for s1, p in v
                }
            )
            for s, v in transition_map.items()
        }
        self.non_terminal_states = list(self.transition_map.keys())

    def __repr__(self) -> str:
        display = ""

        for s, d in self.transition_map.items():
            display += f"From State {s.state}: \n"
            for s1, p in d:
                opt = "Terminal " if isinstance(s1, Terminal) else "State"
                display += f" To {opt} {s1.state} with probability {p:.3f}\n"

        return display

    def transition(self, state: NonTerminal[S]) -> FiniteDistribution[State[S]]:
        return self.transition_map[state]

    def get_transition_matrix(self) -> np.ndarray:
        sz = len(self.non_terminal_states)
        mat = np.zeros((sz, sz))

        for i, s1 in enumerate(self.non_terminal_states):
            for j, s2 in enumerate(self.non_terminal_states):
                mat[i, j] = self.transition(s1).probability(s2)

        return mat

    def get_stationary_distribution(self) -> FiniteDistribution[S]:
        eig_vals, eig_vecs = np.linalg.eig(self.get_transition_matrix().T)
        index_of_first_unit_eig_val = np.where(np.abs(eig_vals - 1) < 1e-8)[0][0]
        eig_vec_of_unit_eig_val = np.real(eig_vecs[:, index_of_first_unit_eig_val])
        return Categorical(
            {
                self.non_terminal_states[i].state: ev
                for i, ev in enumerate(
                    eig_vec_of_unit_eig_val / sum(eig_vec_of_unit_eig_val)
                )
            }
        )

    def display_stationary_distribution(self):
        pprint({s: round(p, 3) for s, p in self.get_stationary_distribution()})

    def generate_image(self) -> graphviz.Digraph:
        d = graphviz.Digraph()

        for s in self.transition_map.keys():
            d.node(str(s))

        for s, v in self.transition_map.items():
            for s1, p in v:
                d.edge(str(s), str(s1), label=str(p))

        return d
