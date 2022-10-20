from torch import nn

class EuropeanOption():
    def __init__(self, call=True, strike=100.0, maturity=30/365, dim=0):
        self.call = call
        self.strike = strike
        self.maturity = maturity  
        self.dim = dim

    def compute_payoff(self, spots):
        if self.call:
            terminal_value = spots[:, self.dim, -1] - self.strike
        else:
            terminal_value = self.strike - spots[:, self.dim, -1]
        return nn.functional.relu(terminal_value)
