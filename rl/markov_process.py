from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Generic,
    Callable,
    Iterable,
    Sequence,
    TypeVar,
    Mapping,
    Set,
    Tuple,
    Dict,
)
from rl.distribution import (
    Categorical,
    Distribution,
    FiniteDistribution,
    SampledDistribution,
)
import numpy as np
from pprint import pprint
import graphviz
from collections import defaultdict

S = TypeVar("S")
X = TypeVar("X")


class State(ABC, Generic[S]):
    """
    The state abstract class that contains the state,
    which could be any type - denoted by Generic S.
    """

    state: S

    def on_non_terminal(self, f: Callable[["NonTerminal[S]"], X], default: X) -> X:
        """
        This function returns whether the Object is a Terminal
        or Non-Terminal State.
        """
        if isinstance(self, NonTerminal):
            return f(self)
        else:
            return default


@dataclass(frozen=True)
class Terminal(State[S]):
    """
    This class contains the Terminal states that inherit from the State base class.
    It is a dataclass so as to accomodate any state of Generic type S.
    """

    state: S


@dataclass(frozen=True)
class NonTerminal(State[S]):
    """
    This class contains the Non Terminal states that inherit from the State base class.
    It is a dataclass so as to accomodate any state of Generic type S.
    """

    state: S

    def __eq__(self, other):
        return self.state == other.state

    def __lt__(self, other):
        return self.state < other.state


class MarkovProcess(ABC, Generic[S]):
    """
    This is the Markov Process Abstract class that inherits a Generic Type as well.
    """

    @abstractmethod
    def transition(self, state: NonTerminal[S]) -> Distribution[State[S]]:
        """
        The transition function returns the transition probability distribution of next
        states, given a current Non Terminal State.
        """
        pass

    def simulate(
        self, start_state_distribution: Distribution[NonTerminal[S]]
    ) -> Iterable[State[S]]:
        """
        Given a state distribution - return a list of all next states and the probability
        distribution for those states by simulating the sample of the distribution.
        """
        state: State[S] = start_state_distribution.sample()
        yield state

        while isinstance(state, NonTerminal):
            state = self.transition(state).sample()
            yield state


# We define Transition as the Mapping between a Non Terminal State
# and the FiniteDistribution of States that succeed it.
Transition = Mapping[NonTerminal[S], FiniteDistribution[State[S]]]


class FiniteMarkovProcess(MarkovProcess[S]):
    """
    The Finite Markov Process contains a finite number of states that have their
    transition probabilities defined within the transition_map.
    """

    non_terminal_states: Sequence[NonTerminal[S]]
    transition_map: Transition[S]

    def __init__(self, transition_map: Mapping[S, FiniteDistribution[S]]):
        """
        We instantiate the FiniteMarkovProcess with just the transition map
        of all states and their transition probabilities as defined by
        the Transition class.

        The keys of the Transition class are the non_terminals as terminal
        states do not have a transition probability function.
        """
        # Get all the non_terminals as a set to remove duplicates.
        # These are the keys to our transition map.
        non_terminals: Set[S] = set(transition_map.keys())
        # Generate the transition_map as a dict of Non Terminal States Mapping to a
        # Categorical Distribution that yields the normalized probabilities for
        # all states that are possible from a given non-terminal state.
        self.transition_map = {
            NonTerminal(s): Categorical(
                {
                    (NonTerminal(s1) if s1 in non_terminals else Terminal(s1)): p
                    for s1, p in v
                }
            )
            for s, v in transition_map.items()
        }
        # The Non Terminal states are now the list of non_terminal states a list of
        # the keys of the transition_map.
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
        """
        The transition function implemented here takes a State as the input
        and yields the FiniteDistribution of the states that can occur afterwards.
        This is yielded by the values of the dictionary transition_map by providing the
        key of state.
        """
        return self.transition_map[state]

    def get_transition_matrix(self) -> np.ndarray:
        """
        For a given non-terminal state, we would like to know the transition probabilities
        as a transition matrix that represents the probability of the next state given
        a starting state.
        This is basically a tabular representation that shows the transition probability
        given row as current state and the column as next state.
        """
        # Instantiate a zero vector of size m - which is the size of all non-terminal states.
        sz = len(self.non_terminal_states)
        mat = np.zeros((sz, sz))

        for i, s1 in enumerate(self.non_terminal_states):
            for j, s2 in enumerate(self.non_terminal_states):
                # At i,j of the vector, assign the transition probability to the
                # matrix as the transition.probability which has been defined in the
                # Distribution class.
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


@dataclass(frozen=True)
class TransitionStep(Generic[S]):
    state: NonTerminal[S]
    next_state: State[S]
    reward: float


class MarkovRewardProcess(MarkovProcess[S]):

    def transition(self, state: NonTerminal[S]) -> Distribution[State[S]]:
        distribution = self.transition_reward(state)

        def next_state(distribution=distribution):
            next_s, _ = distribution.sample()
            return next_s

        return SampledDistribution(next_state)

    @abstractmethod
    def transition_reward(
        self, state: NonTerminal[S]
    ) -> Distribution[Tuple[State[S], float]]:
        pass

    def simulate_reward(
        self, start_state_distribution: Distribution[NonTerminal[S]]
    ) -> Iterable[TransitionStep[S]]:
        state: State[S] = start_state_distribution.sample()
        reward: float = 0

        while isinstance(state, NonTerminal):
            next_distribution = self.transition_reward(state)
            next_state, reward = next_distribution.sample()

            yield TransitionStep(state, next_state, reward)

            state = next_state


StateReward = FiniteDistribution[Tuple[State[S], float]]
RewardTransition = Mapping[NonTerminal[S], StateReward[S]]


class FiniteMarkovRewardProcess(FiniteMarkovProcess[S], MarkovRewardProcess[S]):
    transition_reward_map: RewardTransition[S]
    reward_function_vec: np.ndarray

    def __init__(
        self, transition_reward_map: Mapping[S, FiniteDistribution[Tuple[S, float]]]
    ):
        transition_map: Dict[S, FiniteDistribution[S]] = {}
        for state, trans in transition_reward_map.items():
            probabilities: Dict[S, float] = defaultdict(float)
            for (next_state, _), probability in trans:
                probabilities[next_state] += probability

            transition_map[state] = Categorical(probabilities)

        super().__init__(transition_map)

        nt: Set[S] = set(transition_reward_map.keys())
        self.transition_reward_map = {
            NonTerminal(s): Categorical(
                {
                    (NonTerminal(s1) if s1 in nt else Terminal(s1), r): p
                    for (s1, r), p in v
                }
            )
            for s, v in transition_reward_map.items()
        }

        self.reward_function_vec = np.array(
            [
                sum(
                    probability * reward
                    for (_, reward), probability in self.transition_reward_map[state]
                )
                for state in self.non_terminal_states
            ]
        )

    def __repr__(self) -> str:
        display = ""
        for s, d in self.transition_reward_map.items():
            display += f"From State {s.state}:\n"
            for (s1, r), p in d:
                opt = "Terminal " if isinstance(s1, Terminal) else ""
                display += (
                    f"  To [{opt}State {s1.state} and Reward {r:.3f}]"
                    + f" with Probability {p:.3f}\n"
                )
        return display

    def transition_reward(self, state: NonTerminal[S]) -> StateReward[S]:
        return self.transition_reward_map[state]

    def get_value_function_vec(self, gamma: float) -> np.ndarray:
        return np.linalg.solve(
            np.eye(len(self.non_terminal_states))
            - gamma * self.get_transition_matrix(),
            self.reward_function_vec,
        )

    def display_reward_function(self):
        pprint(
            {
                self.non_terminal_states[i]: round(r, 3)
                for i, r in enumerate(self.reward_function_vec)
            }
        )

    def display_value_function(self, gamma: float):
        pprint(
            {
                self.non_terminal_states[i]: round(v, 3)
                for i, v in enumerate(self.get_value_function_vec(gamma))
            }
        )
