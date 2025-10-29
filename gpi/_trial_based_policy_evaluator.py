from _base import GeneralPolicyIterationComponent
from mdp import ClosedFormMDP
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
        self.gamma = gamma
        self.max_trial_length = max_trial_length
        self.exploring_starts = exploring_starts
        self.random_state = random_state
    
    def step(self):
        """
            creates and processes a trial to update state-values and q-values
        """
        assert self.workspace is not None, "Workspace not set"
        assert callable(self.workspace.policy), "Workspace must hold a policy callable"

        # Generar un trial en DataFrame con columnas ["state","action","reward"]
        mdp = self.trial_interface.mdp
        rs = self.random_state or np.random.RandomState(0)
        rows = []

        # Inicialización: ES o init_states
        if self.exploring_starts:
            s, r = self.trial_interface.get_random_state()   # esta interfaz retorna (s, r)
            # acción aleatoria solo en el primer paso
            acts = self.trial_interface.get_actions_in_state(s)
            if len(acts) == 0:
                rows.append([s, None, r])
                trial = pd.DataFrame(rows, columns=["state","action","reward"])
                report = self.process_trial_for_policy(trial, self.workspace.policy)
                length = int((trial["action"].notna()).sum())
                rep = dict(report or {})
                rep.setdefault("length", length)
                rep.setdefault("exploring_starts", True)
                return rep
            a0 = acts[rs.choice(len(acts))]
            rows.append([s, a0, r])
            s, r = self.trial_interface.exec_action(s, a0)
        else:
            s, r = self.trial_interface.draw_init_state()     # esta interfaz retorna (s, r)

        steps = 0
        while (not mdp.is_terminal_state(s)) and (steps < self.max_trial_length):
            a = self.workspace.policy(s)
            rows.append([s, a, r])
            s, r = self.trial_interface.exec_action(s, a)
            steps += 1

        # Estado terminal sin acción
        rows.append([s, None, r])

        trial = pd.DataFrame(rows, columns=["state","action","reward"])
        report = self.process_trial_for_policy(trial, self.workspace.policy)
        length = int((trial["action"].notna()).sum())
        rep = dict(report or {})
        rep.setdefault("length", length)
        rep.setdefault("exploring_starts", self.exploring_starts)
        return rep
    
    @abstractmethod
    def process_trial_for_policy(self, trial, policy):
        raise NotImplementedError