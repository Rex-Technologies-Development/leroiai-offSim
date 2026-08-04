"""Small TensorBoard callback for Override match outcomes."""
from stable_baselines3.common.callbacks import BaseCallback

class ScoreLoggingCallback(BaseCallback):
    def __init__(self, verbose=0): super().__init__(verbose); self.results=[]
    def _on_step(self):
        for info, done in zip(self.locals.get("infos", []), self.locals.get("dones", [])):
            if done: self.results.append((float(info.get("episode_score",0)), float(info.get("episode_opp_score",0))))
        if len(self.results) >= 20:
            import numpy as np
            values=np.asarray(self.results[-20:]); self.logger.record("game/blue_score",values[:,0].mean()); self.logger.record("game/red_score",values[:,1].mean()); self.logger.record("game/win_rate",(values[:,0]>values[:,1]).mean()); self.results.clear()
        return True
