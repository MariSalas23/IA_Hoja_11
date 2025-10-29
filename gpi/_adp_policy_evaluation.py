from mdp._trial_interface import TrialInterface
import numpy as np

from policy_evaluation._linear import LinearSystemEvaluator
from gpi._trial_based_policy_evaluator import TrialBasedPolicyEvaluator
from mdp._base import ClosedFormMDP

class ADPPolicyEvaluation(TrialBasedPolicyEvaluator):

    def __init__(
            self,
            trial_interface: TrialInterface,
            gamma: float,
            exploring_starts: bool,
            max_trial_length: int = np.inf,
            random_state: np.random.RandomState = None,
            precision_for_transition_probability_estimates=4,
            update_interval: int = 10
        ):
        super().__init__(
            trial_interface=trial_interface,
            gamma=gamma,
            max_trial_length=max_trial_length,
            exploring_starts=exploring_starts,
            random_state=random_state
        )
        self.precision_for_transition_probability_estimates = precision_for_transition_probability_estimates
        self.update_interval = update_interval

        # N(s,a,s') y promedio de recompensas por estado
        self.counts = {}      # dict[s][a][sp] -> int
        self.r_sum = {}       # dict[s] -> float
        self.r_cnt = {}       # dict[s] -> int
        self._steps_seen = 0
    
    def _touch(self, s=None, a=None, sp=None):
        if s is not None:
            if s not in self.counts:
                self.counts[s] = {}
            if s not in self.r_sum:
                self.r_sum[s] = 0.0
                self.r_cnt[s] = 0
            if a is not None:
                if a not in self.counts[s]:
                    self.counts[s][a] = {}
                if sp is not None and sp not in self.counts[s][a]:
                    self.counts[s][a][sp] = 0

    def _build_closed_form_hat(self):
        mdp = self.trial_interface.mdp
        S = list(mdp.states)
        A = list(mdp.actions)
        P = np.zeros((len(S), len(A), len(S)))
        r = np.zeros(len(S), dtype=float)

        # recompensas
        for i, s in enumerate(S):
            r[i] = (self.r_sum[s] / self.r_cnt[s]) if self.r_cnt.get(s, 0) > 0 else 0.0

        # transiciones
        for i, s in enumerate(S):
            for j, a in enumerate(A):
                tot = float(sum(self.counts.get(s, {}).get(a, {}).values()))
                if tot <= 0:
                    continue
                for sp, c in self.counts[s][a].items():
                    k = S.index(sp)
                    P[i, j, k] = round(c / tot, int(self.precision_for_transition_probability_estimates))

        return ClosedFormMDP(S, A, P, r)

    def _evaluate_policy_closed_form(self, mdp_hat, policy):
        S = mdp_hat.states
        A = mdp_hat.actions
        n = len(S)

        # Construir P_pi
        Ppi = np.zeros((n, n))
        for i, s in enumerate(S):
            if mdp_hat.is_terminal_state(s):
                continue
            a = policy(s)
            j = A.index(a)
            Ppi[i, :] = mdp_hat.prob_matrix[i, j, :]

        r = mdp_hat.rewards.astype(float)
        I = np.eye(n)
        try:
            v = np.linalg.solve(I - self.gamma * Ppi, r)
        except np.linalg.LinAlgError:
            # Fallback iterativo
            v = np.zeros(n, dtype=float)
            for _ in range(1000):
                v_next = r + self.gamma * (Ppi @ v)
                if np.max(np.abs(v_next - v)) < 1e-8:
                    v = v_next
                    break
                v = v_next
        return v

    def _push_vq(self, mdp_hat, v_vec):
        # v dict
        v_dict = {s: float(v_vec[i]) for i, s in enumerate(mdp_hat.states)}
        self.workspace.replace_v(v_dict)
        # q dict desde v
        q_hat = mdp_hat.get_q_values_from_v_values(v_vec, self.gamma)
        if self.workspace.q is None:
            self.workspace.replace_q(q_hat)
        else:
            for s, amap in q_hat.items():
                if s not in self.workspace.q:
                    self.workspace.q[s] = {}
                self.workspace.q[s].update(amap)
    
    def process_trial_for_policy(self, df_trial, policy):
        """

        :param df_trial: dataframe with the trial (three columns with states, actions, and the rewards)
        :param policy: the policy that was used to create the trial
        :return: a dictionary with a report of the step
        """
        states = df_trial["state"].tolist()
        actions = df_trial["action"].tolist()
        rewards = df_trial["reward"].tolist()

        # acumular recompensas por estado visitado (al entrar)
        for s, r in zip(states, rewards):
            self._touch(s=s)
            self.r_sum[s] += float(r)
            self.r_cnt[s] += 1

        # acumular transiciones (s,a)->s'
        transitions = 0
        for t in range(len(states) - 1):
            s, a, sp = states[t], actions[t], states[t + 1]
            if a is None:
                continue
            self._touch(s=s, a=a, sp=sp)
            self.counts[s][a][sp] += 1
            transitions += 1

        self._steps_seen += transitions

        recomputed = False
        if (self._steps_seen % int(self.update_interval) == 0) or (self.workspace.q is None):
            mdp_hat = self._build_closed_form_hat()
            v_vec = self._evaluate_policy_closed_form(mdp_hat, policy)
            self._push_vq(mdp_hat, v_vec)
            recomputed = True

        return {
            "transitions": transitions,
            "recomputed_model": recomputed,
            "steps_total": self._steps_seen
        }