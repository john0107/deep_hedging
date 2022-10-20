import torch
from torch import nn

from modules.market_impact import ZeroImpact
from modules.transaction_cost import ZeroCost

class BaseModel():
    def __init__(self, initial_value=100.0, device=torch.device('cpu')):
        self.initial_value = torch.tensor(initial_value, device=device)
        self.device = device

    def get_initial_value(self, n_samples):
        return self.initial_value.repeat(n_samples, 1)

    def generate(self, n_samples, n_dim, n_steps):
        raise NotImplementedError("Please implement this method")
        
    def simulate(self):
        raise NotImplementedError("Please implement this method")


class BaseHedger(nn.Module):

    def __init__(self, 
                 n_dim, 
                 n_steps, 
                 model,
                 market_impact=ZeroImpact(),
                 transaction_cost=ZeroCost(),
                 ) -> None:
        super().__init__()
        self.n_dim = n_dim
        self.n_steps = n_steps

        self.model = model
        assert self.model.initial_value.numel() == self.n_dim

        self.market_impact = market_impact
        self.transaction_cost = transaction_cost
  
    def forward(self, x):
        n_samples, n_dim, n_steps = x.shape
        assert self.n_dim == n_dim and self.n_steps == n_steps

        hedge = x.new_zeros(n_samples, n_dim, n_steps)
        spots = x.new_zeros(n_samples, n_dim, n_steps+1)
        costs = x.new_zeros(n_samples, n_dim, n_steps+1)
        prev_hedge = x.new_zeros(n_samples, n_dim)
        current_spot = self.model.get_initial_value(n_samples)

        for i in range(n_steps):            
            hedge[:, :, i] = self.delta(step=i, spot=current_spot, prev_hedge=prev_hedge)
            spots[:, :, i] = self.market_impact.compute_impacted_price(current_spot, hedge[:, :, i] - prev_hedge)
            costs[:, :, i] = self.transaction_cost.compute_cost(hedge[:, :, i] - prev_hedge, current_spot)

            #with torch.no_grad():
            current_spot = self.model.simulate(spots[:, :, i], x[:, :, i])
            prev_hedge = hedge[:, :, i]

        spots[:, :, n_steps] = self.market_impact.compute_impacted_price(current_spot, -prev_hedge)
        costs[:, :, n_steps] = self.transaction_cost.compute_cost(-prev_hedge, current_spot)

        return hedge, spots, costs
    
    def compute_pnl(self, hedge, spots, costs, derivative=None, price=None):
        PnL = (hedge.mul(spots.diff(dim=-1))).sum(dim=-1)  # compute stochastic integral (\delta \bullet S)_T
        PnL = PnL - costs.sum(dim=-1)  # subtract transaction costs        
        PnL = PnL + self.market_impact.compute_impact_correction(hedge, spots) # add correction due to market impact of trading strategy
        PnL = PnL.sum(dim=1)  # add up PnL for all hedging instruments

        if derivative:
            PnL -= derivative.compute_payoff(spots)  # deduct derivative payoff from PnL
        if price: 
            PnL += price  # add charged price
        return PnL

    def price(self):
        raise NotImplementedError("Please implement this method")

    def delta(self):
        raise NotImplementedError("Please implement this method")
