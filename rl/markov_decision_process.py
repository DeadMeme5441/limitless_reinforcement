from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    DefaultDict,
    Iterable,
    Mapping,
    Sequence,
    Set,
    TypeVar,
    Tuple,
    Dict,
    Generic,
    Iterable,
)
from rl.distribution import (
    Distribution,
    Categorical,
    Constant,
    FiniteDistribution,
    SampledDistribution,
)
from rl.markov_process import (
    MarkovRewardProcess,
    State,
    FiniteMarkovRewardProcess,
    NonTerminal,
    StateReward,
    Terminal,
)
from rl.policy import Policy, FinitePolicy
from collections import defaultdict

A = TypeVar("A")
S = TypeVar("S")

ActionMapping = Mapping[A, StateReward[S]]
StateActionMapping = Mapping[NonTerminal[S], ActionMapping[A, S]]


@dataclass(frozen=True)
class TransitionStep(Generic[S, A]):
    state: NonTerminal[S]
    action: A
    next_state: State[S]
    reward: float


class MarkovDecisionProcess(ABC, Generic[S, A]):
    @abstractmethod
    def actions(self, state: NonTerminal[S]) -> Iterable[A]:
        pass

    @abstractmethod
    def step(
        self, state: NonTerminal[S], action: A
    ) -> Distribution[Tuple[State[S], float]]:
        pass

    def apply_policy(self, policy: Policy[S, A]) -> MarkovRewardProcess[S]:
        mdp = self

        class RewardProcess(MarkovRewardProcess[S]):
            def transition_reward(
                self, state: NonTerminal[S]
            ) -> Distribution[Tuple[State[S], float]]:
                actions: Distribution[A] = policy.act(state)
                return actions.apply(lambda a: mdp.step(state, a))

        return RewardProcess()

    def simulate_actions(
        self, start_states: Distribution[NonTerminal[S]], policy: Policy[S, A]
    ) -> Iterable[TransitionStep[S, A]]:
        state: State[S] = start_states.sample()

        while isinstance(state, NonTerminal):
            action_distribution = policy.act(state)

            action = action_distribution.sample()

            next_distribution = self.step(state, action)

            next_state, reward = next_distribution.sample()

            yield TransitionStep(state, action, next_state, reward)

            state = next_state


class FiniteMarkovDecisionProcess(MarkovDecisionProcess[S, A]):
    mapping: StateActionMapping[S, A]
    non_terminal_states: Sequence[NonTerminal[S]]

    def __init__(
        self, mapping: Mapping[S, Mapping[A, FiniteDistribution[Tuple[S, float]]]]
    ):
        non_terminals: Set[S] = set(mapping.keys())
        self.mapping = {
            NonTerminal(s): {
                a: Categorical(
                    {
                        (NonTerminal(s1) if s1 in non_terminals else Terminal(s1), r): p
                        for (s1, r), p in v
                    }
                )
                for a, v in d.items()
            }
            for s, d in mapping.items()
        }
        self.non_terminal_states = list(self.mapping.keys())

    def __repr__(self) -> str:
        display = ""
        for s, d in self.mapping.items():
            display += f"From State {s.state}: \n"
            for a, d1 in d.items():
                display += f" With Action {a}: \n"
                for (s1, r), p in d1:
                    opt = "Terminal " if isinstance(s1, Terminal) else ""
                    display += (
                        f"   To [{opt}State {s1.state}] and "
                        + f"Reward {r:.3f}] with Probability {p:.3f}\n"
                    )

        return display

    def step(self, state: NonTerminal[S], action: A) -> StateReward[S]:
        action_map: ActionMapping[A, S] = self.mapping[state]
        return action_map[action]

    def actions(self, state: NonTerminal[S]) -> Iterable[A]:
        return self.mapping[state].keys()

    def apply_finite_policy(
        self, policy: FinitePolicy[S, A]
    ) -> FiniteMarkovRewardProcess[S]:
        transition_mapping: Dict[S, FiniteDistribution[Tuple[S, float]]] = {}

        for state in self.mapping:
            action_map: ActionMapping[A, S] = self.mapping[state]
            outcomes: DefaultDict[Tuple[S, float], float] = defaultdict(float)

            actions = policy.act(state)

            for action, p_action in actions:
                for (s1, r), p in action_map[action]:
                    outcomes[(s1.state, r)] += p_action * p

            transition_mapping[state.state] = Categorical(outcomes)

        return FiniteMarkovRewardProcess(transition_mapping)
