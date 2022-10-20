import torch
from torch import nn
from torch.distributions.utils import broadcast_all
from torch.distributions.normal import Normal
import matplotlib.pyplot as plt
from modules.risk_measure import ExpectedValue


def load_model(deep_hedger, file_name, path, **kwargs):
    # load model
    deep_hedger.load_state_dict(torch.load('{}/{}.pth'.format(path, file_name)), **kwargs)

def save_model(deep_hedger, file_name, path, **kwargs):
    # save model
    torch.save(deep_hedger.state_dict(), '{}/{}.pth'.format(path, file_name), **kwargs)

# functions
def show_sample(hedgers, sample=None, model=None, derivative=None, price=None, labels=None):     
    n_dim = hedgers[0].n_dim
    n_steps = hedgers[0].n_steps
    fig, axs = plt.subplots(2, 1, figsize=(10,7), sharex=True)
    if sample is None:
        if model:
            sample = model.generate(1, n_dim, n_steps)
        else: 
            print('Please give a sample or a generator to generate a sample.')
            return
    for i, hedger in enumerate(hedgers):
        if labels:
            label = labels[i]
        else:
            label = hedger.__class__.__name__
        hedger.eval()
        with torch.inference_mode(True):
            hedge, spots, costs = hedger(sample)
            PnL = hedger.compute_pnl(hedge, spots, costs, derivative, price)
        print('{} PnL: {}'.format(label, PnL.item()))
        print('{} Cost: {}'.format(label, costs.sum().item()))
        print('{} Payoff: {}'.format(label, derivative.compute_payoff(spots).item()))
        axs[0].plot(spots.cpu().squeeze(), label=label)
        axs[1].plot(hedge.cpu().squeeze(), label=label)
    axs[0].set_title('Spot Price')
    axs[1].set_title('Hedging Strategy')
    axs[1].legend(bbox_to_anchor=(1,1), loc='lower left')
    plt.show()

def compare_hedgers(hedgers, model, derivative=None, price=None, risk_measure=None, labels=None, table_labels=None, n_samples=100000, table=False, file_name=None, **kwargs):
    n_dim = hedgers[0].n_dim
    n_steps = hedgers[0].n_steps 
    x = model.generate(n_samples, n_dim, n_steps)
    data, label_list = [], []
    table_data = [[] for _ in hedgers]

    rms = [ExpectedValue()]
    if isinstance(risk_measure, list):
        rms += risk_measure
        if table and not table_labels:
            table_labels = [rm.__class__.__name__ for rm in risk_measure]
    elif risk_measure:
        rms += [risk_measure]
        if table and not table_labels:
            table_labels = [risk_measure.__class__.__name__]

    for i, hedger in enumerate(hedgers):
        if labels: label = labels[i]
        else: label = hedger.__class__.__name__
        if isinstance(price, list): pr = price[i]
        else: pr = price
        hedger.eval()
        with torch.inference_mode(True):
            hedge, spots, costs = hedger(x)
            PnL = hedger.compute_pnl(hedge, spots, costs, derivative=derivative, price=pr)
            for rm in rms:
                table_data[i].append(round(rm.compute_risk(PnL).item(), 4))
      
        data.append(PnL.numpy()) 
        label_list.append(label)

    # plot histogram
    plt.hist(data, label=label_list, bins=30, **kwargs)
    plt.legend(loc='upper left')

    # plot table       
    if table:    
        column_labels = ['Mean Loss'] + table_labels
        plt.table(cellText=table_data, colLabels=column_labels, rowLabels=label_list, 
                  cellLoc='center', colLoc='center', rowLoc='center', bbox=(0, -0.65, 1, 0.5))
        plt.draw()
    if file_name:
        plt.savefig(file_name,  bbox_inches="tight")
    plt.show()

def compare_to_hedger(hedger, 
                     hedger_class,
                     derivative,
                     price=None,
                     risk_measure=None,
                     n_samples=100000, 
                     labels=None,
                     table_labels=None,
                     table=False,    
                     file_name=None,                 
                     device=torch.device('cpu'),
                     **kwargs):
    assert hedger.n_dim == 1
    n_dim = hedger.n_dim
    n_steps = hedger.n_steps
    model = hedger.model
    market_impact = hedger.market_impact
    transaction_cost = hedger.transaction_cost

    model_hedger = hedger_class(n_dim=n_dim, n_steps=n_steps, model=model, 
                                market_impact=market_impact, transaction_cost=transaction_cost, 
                                **kwargs).to(device)

    compare_hedgers([hedger, model_hedger], model=model, derivative=derivative, price=price, 
                    risk_measure=risk_measure, n_samples=n_samples,
                    labels=labels, table_labels=table_labels, table=table, file_name=file_name)

def compare_strategy(hedgers, n_step, labels=None, start=90, end=110, steps=50, file_name=None):
    grid = torch.linspace(start, end, steps)[:, None]
    for i, hedger in enumerate(hedgers):
        if labels:
            label = labels[i]
        else:
            label = hedger.__class__.__name__
        hedger.eval()
        with torch.inference_mode(True):
            strategy = hedger.delta(step=n_step, spot=grid)
        plt.plot(grid.cpu(), strategy.cpu(), label=label)
    plt.legend(loc='upper left')
    if file_name:
        plt.savefig(file_name)
    plt.show()

# BS functions
def ncdf(x):
    return Normal(0.0, 1.0).cdf(x)

def npdf(x):
    return Normal(0.0, 1.0).log_prob(x).exp()

def d1(log_moneyness, time_to_maturity, volatility):
    s, t, v = broadcast_all(log_moneyness, time_to_maturity, volatility)
    return (s + (v.square() / 2) * t).div(v * t.sqrt())

def d2(log_moneyness, time_to_maturity, volatility):
    s, t, v = broadcast_all(log_moneyness, time_to_maturity, volatility)
    return (s - (v.square() / 2) * t).div(v * t.sqrt())
