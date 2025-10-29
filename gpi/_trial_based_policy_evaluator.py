from _base import GeneralPolicyIterationComponent
from mdp._trial_interface import TrialInterface
import numpy as np
import pandas as pd
from abc import abstractmethod


class TrialBasedPolicyEvaluator(GeneralPolicyIterationComponent):

    def __init__(
            self,
            trial_interface: TrialInterface,
            gamma: float,
            exploring_starts: bool,
            max_trial_length: int = np.inf,
            random_state: np.random.RandomState = None
        ):
        super().__init__()
        self.trial_interface = trial_interface
        self.gamma = float(gamma)
        self.max_trial_length = max_trial_length
        self.exploring_starts = bool(exploring_starts)
        self.rng = random_state or np.random.RandomState(0)
        self.last_trial = None

    def _only_state(self, x):
        # supports x being either state or (state, reward)
        if isinstance(x, (tuple, list)) and len(x) >= 1:
            return x[0]
        return x

    def _maybe_reward(self, x):
        # returns reward if x is (state, reward); else None
        if isinstance(x, (tuple, list)) and len(x) >= 2:
            try:
                return float(x[1])
            except Exception:
                return None
        return None

    def _reward_of(self, state):
        get_r = getattr(self.trial_interface, "get_reward", None)
        if callable(get_r):
            try:
                return float(get_r(state))
            except Exception:
                pass
        return 0.0

    def _rollout(self, policy):
        rows = []

        # initial state (with or without exploring starts)
        if self.exploring_starts:
            ret = self.trial_interface.get_random_state()
            s = self._only_state(ret)
            r0 = self._maybe_reward(ret)
            if r0 is None:
                r0 = self._reward_of(s)
            avail = self.trial_interface.get_actions_in_state(s)
            if len(avail) == 0:
                rows.append([s, None, r0])
                return pd.DataFrame(rows, columns=["state", "action", "reward"])
            a0 = avail[self.rng.choice(len(avail))]
            rows.append([s, a0, r0])
            s, r = self.trial_interface.exec_action(s, a0)
        else:
            ret = self.trial_interface.draw_init_state()
            s = self._only_state(ret)
            r = self._maybe_reward(ret)
            if r is None:
                r = self._reward_of(s)

        # iterate up to max length
        steps = 0
        while steps < self.max_trial_length:
            avail = self.trial_interface.get_actions_in_state(s)
            if len(avail) == 0:
                rows.append([s, None, r])
                break

            a = policy(s)
            rows.append([s, a, r])
            s, r = self.trial_interface.exec_action(s, a)
            steps += 1

        # ensure a terminal row exists if loop ended by length
        if len(rows) == 0 or rows[-1][1] is not None:
            avail = self.trial_interface.get_actions_in_state(s)
            if len(avail) == 0:
                rows.append([s, None, r])

        return pd.DataFrame(rows, columns=["state", "action", "reward"])

    def step(self):
        """
            creates and processes a trial to update state-values and q-values
        """
        assert self.workspace is not None, "Workspace not set."
        assert callable(self.workspace.policy), "A policy must be set in workspace."

        trial_df = self._rollout(self.workspace.policy)
        self.last_trial = trial_df

        report = self.process_trial_for_policy(trial_df, self.workspace.policy) or {}
        report.setdefault("trial_length", int(len(trial_df)))
        report.setdefault("length", int((trial_df["action"].notna()).sum()))
        report.setdefault("exploring_starts", self.exploring_starts)
        return report

    @abstractmethod
    def process_trial_for_policy(self, trial, policy):
        raise NotImplementedError