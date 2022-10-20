import torch


class ZeroImpact():
    def compute_impacted_price(self, spot, hedge, **kwargs):
        return spot

    def compute_impact_correction(self, hedge, *args, **kwargs):
        n_samples, n_dim, n_steps = hedge.shape
        return hedge.new_zeros(n_samples, n_dim)

class FixedImpact():
    def __init__(self, lambd=1.0):
        self.lambd = lambd

    def compute_impacted_price(self, spot, hedge, **kwargs):
        return spot + self.lambd * hedge
      
    def compute_impact_correction(self, hedge, *args, **kwargs):
        n_samples, n_dim, n_steps = hedge.shape
        zeros = hedge.new_zeros(n_samples, n_dim, 1)
        return 1/2 * self.lambd * hedge.diff(dim=-1, prepend=zeros, append=zeros).square().sum(dim=-1)


class SquareRootImpact():
    def __init__(self, lambd=1.0):
        self.lambd = lambd

    def compute_impacted_price(self, spot, hedge, **kwargs):
        return spot + self.lambd * hedge.sign() * hedge.abs().sqrt()

    def compute_impact_correction(self, hedge, *args, **kwargs):
        n_samples, n_dim, n_steps = hedge.shape
        zeros = hedge.new_zeros(n_samples, n_dim, 1)
        return 1/3 * self.lambd * hedge.diff(dim=-1, prepend=zeros, append=zeros).abs().pow(3/2).sum(dim=-1)
