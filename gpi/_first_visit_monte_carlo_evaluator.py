import numpy as np

from mdp._trial_interface import TrialInterface
from gpi._trial_based_policy_evaluator import TrialBasedPolicyEvaluator


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
        # contadores para promedio incremental
        self._counts = {}   # dict[s][a] -> int

    def _ensure_slots(self, s, a):
        if self.workspace.q is None:
            self.workspace.replace_q({})
        if s not in self.workspace.q:
            self.workspace.q[s] = {}
        if a not in self.workspace.q[s]:
            self.workspace.q[s][a] = 0.0
        if s not in self._counts:
            self._counts[s] = {}
        if a not in self._counts[s]:
            self._counts[s][a] = 0

    def _sync_v_from_q(self):
        pi = self.workspace.policy
        mdp = self.trial_interface.mdp
        v = {}
        for s in mdp.states:
            if mdp.is_terminal_state(s):
                v[s] = mdp.get_reward(s)
            else:
                a = pi(s)
                v[s] = self.workspace.q.get(s, {}).get(a, 0.0)
        self.workspace.replace_v(v)

    def process_trial_for_policy(self, df_trial, policy):
        """

        :param df_trial: dataframe with the trial (three columns with states, actions, and the rewards)
        :param policy: the policy that was used to create the trial
        :return: a dictionary with a report of the step
        """
        states = df_trial["state"].tolist()
        actions = df_trial["action"].tolist()
        rewards = df_trial["reward"].tolist()

        # Retornos descontados hacia atrás
        G = 0.0
        ret = [0.0] * len(rewards)
        for i in range(len(rewards) - 1, -1, -1):
            G = rewards[i] + self.gamma * G
            ret[i] = G

        # First-Visit: solo la primera ocurrencia de (s,a) en el trial
        seen = set()
        updates = 0
        for t, (s, a) in enumerate(zip(states, actions)):
            if a is None:
                continue
            key = (s, a)
            if key in seen:
                continue
            seen.add(key)

            # La acción afecta desde s_{t+1} en adelante:
            target = ret[t + 1] if (t + 1) < len(ret) else 0.0

            self._ensure_slots(s, a)
            self._counts[s][a] += 1
            n = float(self._counts[s][a])
            old = self.workspace.q[s][a]
            self.workspace.q[s][a] = old + (target - old) / n
            updates += 1

        self._sync_v_from_q()
        return {"updated_pairs": updates}