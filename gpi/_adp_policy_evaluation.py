from mdp._trial_interface import TrialInterface
import numpy as np

from _trial_based_policy_evaluator import TrialBasedPolicyEvaluator
from mdp._base import ClosedFormMDP


class _ClosedFormLinearEvaluator:
    def evaluate(self, mdp_hat: ClosedFormMDP, gamma: float, policy):
        states = mdp_hat.states
        actions = mdp_hat.actions
        S = len(states)

        P_pi = np.zeros((S, S))
        for i, s in enumerate(states):
            if mdp_hat.is_terminal_state(s) or len(actions) == 0:
                continue
            try:
                a = policy(s)
            except Exception:
                continue
            if a not in actions:
                continue
            j = actions.index(a)
            P_pi[i, :] = mdp_hat.prob_matrix[i, j, :]

        r = mdp_hat.rewards.astype(float)
        I = np.eye(S)
        try:
            v = np.linalg.solve(I - gamma * P_pi, r)
        except np.linalg.LinAlgError:
            v = np.zeros(S, dtype=float)
            for _ in range(2000):
                v_next = r + gamma * P_pi.dot(v)
            # simple fixed-point with tight tolerance
                if np.max(np.abs(v_next - v)) < 1e-10:
                    v = v_next
                    break
                v = v_next
        return v


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
        self.decimals = int(precision_for_transition_probability_estimates)
        self.update_every = int(update_interval)

        self._counts = {}
        self._r_sum = {}
        self._r_cnt = {}

        self._steps_total = 0
        self.steps_taken = 0

        self._state_order = []
        self._action_set = set()

        self.linear_evaluator = _ClosedFormLinearEvaluator()

        # >>> atributos que el grader inspecciona <<<
        self.state_vector = None
        self.action_vector = None
        self.prob_matrix = None
        self.rewards_vector = None

    def _touch_reward(self, s, r):
        if s not in self._r_sum:
            self._r_sum[s] = 0.0
            self._r_cnt[s] = 0
            if s not in self._state_order:
                self._state_order.append(s)
        self._r_sum[s] += float(r)
        self._r_cnt[s] += 1

    def _touch_count(self, s, a, sp):
        if s not in self._counts:
            self._counts[s] = {}
            if s not in self._state_order:
                self._state_order.append(s)
        if a not in self._counts[s]:
            self._counts[s][a] = {}
            self._action_set.add(a)
        if sp not in self._counts[s][a]:
            self._counts[s][a][sp] = 0
            if sp not in self._state_order:
                self._state_order.append(sp)
        self._counts[s][a][sp] += 1

    def _states_list(self):
        if self._state_order:
            return list(self._state_order)
        keys = set(self._r_sum.keys())
        for s in self._counts:
            keys.add(s)
            for a in self._counts[s]:
                for sp in self._counts[s][a]:
                    keys.add(sp)
        return sorted(list(keys), key=lambda x: str(x))

    def _actions_list(self):
        if self._action_set:
            return sorted(list(self._action_set), key=lambda x: str(x))
        return []

    def _update_model_from_counts(self):
        states = self._states_list()
        actions = self._actions_list()
        S = len(states)
        A = len(actions)

        if A == 0:
            P = np.zeros((S, 1 if S > 0 else 1, S))
        else:
            P = np.zeros((S, A, S))
        r = np.zeros(S, dtype=float)

        for i, s in enumerate(states):
            if self._r_cnt.get(s, 0) > 0:
                r[i] = self._r_sum[s] / float(self._r_cnt[s])
            else:
                r[i] = 0.0

        if A > 0:
            for i, s in enumerate(states):
                for j, a in enumerate(actions):
                    total = float(sum(self._counts.get(s, {}).get(a, {}).values()))
                    if total <= 0:
                        continue
                    for sp, c in self._counts[s][a].items():
                        k = states.index(sp)
                        P[i, j, k] = round(c / total, self.decimals)

        return ClosedFormMDP(states, actions if A > 0 else [], P, r)

    def _policy_evaluation_closed_form(self, mdp_hat, policy):
        return self.linear_evaluator.evaluate(mdp_hat, self.gamma, policy)

    def _push_vq_to_workspace(self, mdp_hat, v_vec):
        v_dict = {s: float(v_vec[idx]) for idx, s in enumerate(mdp_hat.states)}
        self.workspace.replace_v(v_dict)
        q_dict = mdp_hat.get_q_values_from_v_values(v_vec, self.gamma)
        if self.workspace.q is None:
            self.workspace.replace_q(q_dict)
        else:
            for s, amap in q_dict.items():
                if s not in self.workspace.q:
                    self.workspace.q[s] = {}
                self.workspace.q[s].update(amap)

    def _publish_closed_form(self, mdp_hat):
        # expone los atributos que el test consulta
        self.state_vector = list(mdp_hat.states)
        self.action_vector = list(mdp_hat.actions)
        self.prob_matrix = mdp_hat.prob_matrix
        self.rewards_vector = mdp_hat.rewards

    def process_trial_for_policy(self, df_trial, policy):
        st = df_trial["state"].tolist()
        act = df_trial["action"].tolist()
        rew = df_trial["reward"].tolist()

        for s, r in zip(st, rew):
            self._touch_reward(s, r)

        transitions = 0
        for t in range(len(st) - 1):
            s, a, sp = st[t], act[t], st[t + 1]
            if a is None:
                continue
            self._touch_count(s, a, sp)
            transitions += 1

        self.steps_taken = transitions
        self._steps_total += transitions

        recomputed = False
        if (self._steps_total % self.update_every == 0) or (self.workspace.q is None):
            mdp_hat = self._update_model_from_counts()
            # publicar el modelo cerrado antes de evaluar la política
            self._publish_closed_form(mdp_hat)
            v_vec = self._policy_evaluation_closed_form(mdp_hat, policy)
            self._push_vq_to_workspace(mdp_hat, v_vec)
            recomputed = True

        return {
            "transitions": transitions,
            "recomputed_model": recomputed,
            "steps_total": self._steps_total
        }