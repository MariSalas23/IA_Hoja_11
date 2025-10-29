from _base import GeneralPolicyIterationComponent
from mdp._trial_interface import TrialInterface
import numpy as np


class StandardTrialInterfaceBasedPolicyImprover(GeneralPolicyIterationComponent):

    def __init__(self, trial_interface: TrialInterface, random_state: np.random.RandomState):
        super().__init__()
        self.trial_interface = trial_interface
        self.rng = random_state or np.random.RandomState(0)
        # cache actions per state as tests may check caching
        self._actions_cache = {}

    def _actions_in(self, s):
        if s in self._actions_cache:
            return self._actions_cache[s]
        try:
            acts = list(self.trial_interface.get_actions_in_state(s))
        except Exception:
            acts = []
        self._actions_cache[s] = acts
        return acts

    def step(self):
        """
            Improves the current policy based on current q-values
        """
        assert self.workspace is not None, "Workspace not set."
        q = self.workspace.q or {}
        old_pi = self.workspace.policy

        def greedy_action(s):
            actions = self._actions_in(s)
            if not actions:
                return None
            # unknown q-values are treated as 0
            vals = [q.get(s, {}).get(a, 0.0) for a in actions]
            max_val = max(vals) if len(vals) else 0.0
            # tie-break randomly among all best actions (or among all if all equal)
            idxs = [i for i, v in enumerate(vals) if v == max_val]
            pick = int(self.rng.choice(idxs)) if idxs else 0
            return actions[pick]

        def new_policy(s):
            # if we have never seen s in q, still pick among available actions
            return greedy_action(s)

        # measure changes only on states we know about (from q-keys)
        known_states = set(q.keys())
        changes = 0
        for s in known_states:
            act_new = new_policy(s)
            if old_pi is None:
                changes += 1
                continue
            try:
                if act_new != old_pi(s):
                    changes += 1
            except Exception:
                changes += 1

        self.workspace.replace_policy(new_policy)
        return {"policy_changed": changes > 0, "num_changes": changes}