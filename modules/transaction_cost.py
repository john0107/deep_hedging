import torch


class ZeroCost():
    def compute_cost(self, hedge, spot):
        return torch.zeros_like(hedge)


class FixedCost():
    def __init__(self, c=0.1):
        self.c = c
      
    def compute_cost(self, hedge, spot):
        return self.c * hedge.abs() * spot 

class ProportionalCost():
    def __init__(self, c=0.1):
        self.c = c
      
    def compute_cost(self, hedge, spot):
        return self.c * hedge.abs() * spot