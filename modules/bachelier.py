import torch
from torch import nn
from torch.distributions.utils import broadcast_all

from scipy.optimize import fixed_point, newton
from scipy.stats import norm

from modules.utils import *
from modules.derivative import EuropeanOption
from modules.base_model import BaseModel, BaseHedger


# Bachelier generator and simulator
class BachelierModel(BaseModel):
    def __init__(self, mu=0.0, sigma=0.2, dt=1/365, **kwargs):
        super().__init__(**kwargs)
        self.mu = mu
        self.sigma = sigma
        self.dt = dt

    def generate(self, n_samples, n_dim, n_steps):
        x = torch.randn((n_samples, n_dim, n_steps), device=self.device)
        x = x * self.sigma * torch.tensor(self.dt, device=self.device).sqrt()
        x = self.initial_value * (x + self.mu * torch.tensor(self.dt, device=self.device))
        return x
        
    def simulate(self, spot, increment):
        return spot + increment


class BachelierHedger(BaseHedger):

    def __init__(self, 
                 n_dim=1,
                 n_steps=30, 
                 model=BachelierModel(),
                 derivative=EuropeanOption(),               
                 sigma=0.2,
                 **kwargs):
        '''
        Parameters
        ----------
        n_dim: int, default=1 (only n_dim=1 supported for now)
            market dimension
        n_steps: int, default=30
            time discretization
        model: float, default=BachelierModel
            model to simulate prices from generated data
        
        maturity: float, default=30/365
            maturity of option to hedge      
        sigma: float, default=0.2
            (implied) volatility of the Bachelier model
        strike: float, default=100.0
            strike of the option
        call: boolean, default=True
            whether the option is a (european) call or put option

        Optional Parameters
        -------------------
        market_impact: default=ZeroImpact()
            market impact model
        transaction_cost: default=ZeroCost()
            transaction cost model
        '''
        super().__init__(n_dim=n_dim, 
                         n_steps=n_steps, 
                         model=model, 
                         **kwargs)
        assert isinstance(derivative, EuropeanOption)
        self.maturity = derivative.maturity
        self.strike = derivative.strike
        self.call = derivative.call
        self.sigma = sigma

    def price(self, spot=None, strike=None, time_to_maturity=None):
        if spot is None:
            spot = self.model.initial_value
        if strike is None:
            strike = self.strike
        if time_to_maturity is None:
            time_to_maturity = self.maturity
        s, k, t, v = broadcast_all(spot, strike, time_to_maturity, self.sigma*self.model.initial_value)
        normalized_moneyness = (s - k).div(v*t.sqrt())
        price = (s-k)*ncdf(normalized_moneyness) + v*t.sqrt()*npdf(normalized_moneyness)
        if not self.call:
            price += k - s # put-call parity
        return price

    def delta(self, step=None, spot=None, **kwargs):
        if spot is None:
            spot = self.model.initial_value
        if not 'strike' in kwargs:
            strike = self.strike
        if step:
            time_to_maturity = self.maturity - step * self.maturity / self.n_steps
        else:
            time_to_maturity = self.maturity
        s, k, t, v = broadcast_all(spot, strike, time_to_maturity, self.sigma*self.model.initial_value)
        normalized_moneyness = (s - k).div(v*t.sqrt())
        delta = ncdf(normalized_moneyness)
        return delta if self.call else 1 - delta


class BachelierHedgerFixedImpact(BachelierHedger):
    def __init__(self, 
                 lambd=1.0,
                 **kwargs
                 ) -> None:
        super().__init__(**kwargs)
        self.lambd = lambd

    def delta(self, step=None, spot=None, **kwargs):
        if spot is None:
            spot = self.model.initial_value
        if not 'strike' in kwargs:
            strike = self.strike
        if step:
            time_to_maturity = self.maturity - step * self.maturity / self.n_steps
        else:
            time_to_maturity = self.maturity
        s, k, t, v = spot, strike, time_to_maturity, self.sigma*self.model.initial_value
        x0 = torch.zeros_like(s).numpy()
        try:  # use fixed point iteration for finding fixed point
            def func_fixedpoint(delta):
                return norm.cdf(-self.lambd*delta, k - s, v*t**0.5)
            delta = torch.from_numpy(fixed_point(func_fixedpoint, x0, xtol=1e-08, maxiter=500))
        except:  # use newton if fixed point iteration does not converge
            def func_newton(delta):
                return norm.cdf(-self.lambd*delta, k - s, v*t**0.5) - delta
            def fprime_newton(delta):
                return (-self.lambd / (v*t**0.5)) * norm.pdf(-self.lambd*delta, k - s, v*t**0.5) - 1
            try:
                delta = torch.from_numpy(newton(func_newton, x0, fprime_newton, tol=1.0e-07, maxiter=500))
            except: # use halley if newton iteration does not converge
                def fprime2_newton(delta):
                    return (s - delta * self.lambd - k) / (v**2 * t) * self.lambd * norm.pdf(-self.lambd*delta, k - s, v*t**0.5)
                delta = torch.from_numpy(newton(func_newton, x0, fprime=fprime_newton, fprime2=fprime2_newton, tol=1.0e-06, maxiter=500))

        return delta if self.call else 1 - delta


def compare_to_bachelier_hedger(hedger, derivative, sigma, **kwargs):
    compare_to_hedger(hedger, BachelierHedger, derivative, sigma=sigma, **kwargs)

def compare_to_bachelier_hedger_fixed_impact(hedger, derivative, sigma, lambd, **kwargs):
    compare_to_hedger(hedger, BachelierHedgerFixedImpact, derivative, sigma=sigma, lambd=lambd, **kwargs)
