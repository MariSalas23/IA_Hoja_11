import numpy as np
from mdp._trial_interface import TrialInterface
from _trial_based_policy_evaluator import TrialBasedPolicyEvaluator


class FirstVisitMonteCarloEvaluator(TrialBasedPolicyEvaluator):

    def __init__(
            self,
            trial_interface: TrialInterface,
            gamma: float,
            exploring_starts: bool,
            max_trial_length: int = np.inf,
            random_state: np.random.RandomState = None
        ):
        super().__init__(
            trial_interface=trial_interface,
            gamma=gamma,
            exploring_starts=exploring_starts,
            max_trial_length=max_trial_length,
            random_state=random_state
        )
        # first-visit counts per (s,a)
        self._visit_counts = {}
        # keep a running set of seen states to define v over them
        self._seen_states = set()

    def _ensure_q(self, s, a):
        if self.workspace.q is None:
            self.workspace.replace_q({})
        if s not in self.workspace.q:
            self.workspace.q[s] = {}
        if a not in self.workspace.q[s]:
            self.workspace.q[s][a] = 0.0
        if s not in self._visit_counts:
            self._visit_counts[s] = {}
        if a not in self._visit_counts[s]:
            self._visit_counts[s][a] = 0

    def _is_terminal(self, s):
        try:
            return len(self.trial_interface.get_actions_in_state(s)) == 0
        except Exception:
            return False

    def _reward_of(self, s):
        get_r = getattr(self.trial_interface, "get_reward", None)
        if callable(get_r):
            try:
                return float(get_r(s))
            except Exception:
                pass
        return 0.0

    def _update_v_from_q(self):
        # v(s) = q(s, pi(s)) for deterministic policy; define only for seen states
        pi = self.workspace.policy
        v = {}
        for s in self._seen_states:
            if self._is_terminal(s):
                v[s] = self._reward_of(s)
                continue
            a = pi(s)
            v[s] = self.workspace.q.get(s, {}).get(a, 0.0)
        self.workspace.replace_v(v)

    def process_trial_for_policy(self, df_trial, policy):
        states = df_trial["state"].tolist()
        actions = df_trial["action"].tolist()
        rewards = df_trial["reward"].tolist()

        # remember all states touched in this trial
        for s in states:
            self._seen_states.add(s)

        # returns-from-right
        G = 0.0
        returns_from = [0.0] * len(rewards)
        for i in range(len(rewards) - 1, -1, -1):
            G = rewards[i] + self.gamma * G
            returns_from[i] = G

        # first-visit update
        seen = set()
        updates = 0
        for t, (s, a) in enumerate(zip(states, actions)):
            if a is None:
                continue
            key = (s, a)
            if key in seen:
                continue
            seen.add(key)

            # use return from the NEXT index (effect of (s,a) impacts next state onward)
            target = returns_from[t + 1] if (t + 1) < len(returns_from) else 0.0

            self._ensure_q(s, a)
            self._visit_counts[s][a] += 1
            n = self._visit_counts[s][a]
            old = self.workspace.q[s][a]
            self.workspace.q[s][a] = old + (target - old) / float(n)
            updates += 1

        self._update_v_from_q()
        return {"updated_pairs": updates}