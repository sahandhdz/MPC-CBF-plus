import numpy as np

class SystemModel():
    """
    Base class for discrete-time system models.
    """
    def __init__(
            self,
            x0: np.ndarray,
            n_x: int,
            n_u: int,
            dt: float,
            x_min: np.ndarray,
            x_max: np.ndarray,
            u_min: np.ndarray,
            u_max: np.ndarray,
            dynamics_fn=None
    ):
        self.x0 = x0.copy() # initial state
        self.x = x0.copy()  # internal state
        self.n_x = n_x
        self.n_u = n_u
        self.dt = dt
        self.x_min = x_min
        self.x_max = x_max
        self.u_min = u_min
        self.u_max = u_max
        self.dynamics_fn = dynamics_fn

    def step(self, u: np.ndarray) -> np.ndarray:
        """
        Advance the system by one step: x_{k+1} = f(x_k, u_k).
        Updates internal state and returns the new state.
        """
        if self.dynamics_fn is not None:
            self.x = self.dynamics_fn(self.x, u)
        else:
            raise NotImplementedError("Either pass dynamics_fn or subclass SystemModel and override the step() method")
        return self.x
    
    def reset(self) -> np.ndarray:
        """Restore the model to its initial state x0."""
        self.x = self.x0.copy()
        return self.x