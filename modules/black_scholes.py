import torch
from torch import nn
from torch.distributions.normal import Normal
from torch.distributions.utils import broadcast_all

from modules.utils import *
from modules.derivative import EuropeanOption
from modules.base_model import BaseModel, BaseHedger


# Black-Scholes generator and simulator
class BlackScholesModel(BaseModel):
    def __init__(self, mu=0.0, sigma=0.2, dt=1/365, **kwargs):
        super().__init__(**kwargs)
        self.mu = mu
        self.sigma = sigma
        self.dt = dt

    def generate(self, n_samples, n_dim, n_steps):
        x = torch.randn((n_samples, n_dim, n_steps), device=self.device)
        x = x * self.sigma * torch.tensor(self.dt, device=self.device).sqrt()
        x = x - (self.sigma ** 2) * torch.tensor(self.dt, device=self.device) / 2 
        return x.exp()
        
    def simulate(self, spot, x):
        return spot * x


class BlackScholesHedger(BaseHedger):

    def __init__(self, 
                 n_dim=1,
                 n_steps=30, 
                 model=BlackScholesModel(),
                 derivative=EuropeanOption(),
                 sigma=0.2,
                 **kwargs
                 ) -> None:
        super().__init__(n_dim=n_dim, 
                         n_steps=n_steps, 
                         model=model, 
                         **kwargs)
        assert isinstance(derivative, EuropeanOption)
        self.maturity = derivative.maturity
        self.strike = derivative.strike
        self.call = derivative.call
        self.sigma = sigma

    def price(self, log_moneyness=None, time_to_maturity=None):
        if log_moneyness is None:
            log_moneyness = (self.model.initial_value/self.strike).log()
        if time_to_maturity is None:
            time_to_maturity = self.maturity
        s, t, v = broadcast_all(log_moneyness, time_to_maturity, self.sigma)

        n1 = ncdf(d1(s, t, v))
        n2 = ncdf(d2(s, t, v))

        price = self.strike * (s.exp() * n1 - n2)

        if not self.call:
            price += self.strike * (1 - s.exp())  # put-call parity

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
        log_moneyness = (spot / strike).log()
        volatility = self.sigma
        delta = ncdf(d1(log_moneyness, time_to_maturity, volatility))
        return delta if self.call else 1 - delta


def compare_to_black_scholes_hedger(hedger, derivative, sigma, **kwargs):
    compare_to_hedger(hedger, BlackScholesHedger, derivative, sigma=sigma, **kwargs)
