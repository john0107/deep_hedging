'''Risk Measures'''
from math import ceil


class WorstCaseRiskMeasure():
    def compute_risk(self, PnL):
        return -PnL.min()

class NeutralRiskMeasure():
    def compute_risk(self, PnL):
        return -PnL.mean()


class ExpectedValue():
    def compute_risk(self, PnL):
        return PnL.mean()        


class QuadraticHedgingError():
    def compute_risk(self, PnL):
        return PnL.square().mean()


class AverageValueAtRisk():
    def __init__(self, alpha=0.5):
        assert 0 <= alpha <= 1
        self.alpha = alpha

    def compute_risk(self, PnL):
        return -PnL.topk(ceil(self.alpha * PnL.numel()), dim=0, largest=False).values.mean(dim=0)


class EntropicRiskMeasure():
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def compute_risk(self, PnL):
        return PnL.mul(-self.alpha).exp().mean(dim=0).log().div(self.alpha)
