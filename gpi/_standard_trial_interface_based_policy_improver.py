from ._base import GeneralPolicyIterationComponent
from mdp._trial_interface import TrialInterface
import numpy as np


class StandardTrialInterfaceBasedPolicyImprover(GeneralPolicyIterationComponent):

    def __init__(self, trial_interface: TrialInterface, random_state: np.random.RandomState):
        super().__init__()
        self.trial_interface = trial_interface
        self.random_state = random_state
    
    def step(self):
        """
            Improves the current policy based on current q-values
        """
        assert self.workspace is not None, "Workspace not set"
        rs = self.random_state or np.random.RandomState(0)
        q = self.workspace.q or {}
        old_pi = self.workspace.policy

        def greedy(s):
            acts = self.trial_interface.get_actions_in_state(s)
            if not acts:
                return None
            vals = [q.get(s, {}).get(a, 0.0) for a in acts]
            m = max(vals)
            idx = [i for i, v in enumerate(vals) if v == m]
            return acts[rs.choice(idx)]

        def new_pi(s):
            return greedy(s)

        # contar cambios frente a la política actual
        changes = 0
        for s in self.trial_interface.mdp.states:
            if self.trial_interface.mdp.is_terminal_state(s):
                continue
            if old_pi is None:
                changes += 1
            else:
                try:
                    if new_pi(s) != old_pi(s):
                        changes += 1
                except Exception:
                    changes += 1

        self.workspace.replace_policy(new_pi)
        return {"policy_changed": changes > 0, "num_changes": changes}