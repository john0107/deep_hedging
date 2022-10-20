import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm, trange # another option would be fastprogress from fastai

from modules.base_model import BaseHedger


class NeuralNetwork(nn.Module):
    def __init__(self, 
                 n_input: int = 1, 
                 n_output: int = 1, 
                 n_layers: int = 3, 
                 n_nodes: int = 16) -> None:
        super(NeuralNetwork, self).__init__()
        self.n_input: int = n_input
        self.n_output: int = n_output
        self.n_layers: int = n_layers
        self.n_nodes: int = n_nodes
        linear_relu_stack = [nn.Linear(n_input, n_nodes, bias=False), nn.BatchNorm1d(n_nodes), nn.ReLU()]
        for _ in range(1, n_layers-1):
            linear_relu_stack += [nn.Linear(n_nodes, n_nodes, bias=False), nn.BatchNorm1d(n_nodes), nn.ReLU()]
        linear_relu_stack += [nn.Linear(n_nodes, n_output)]
        self.linear_relu_stack = nn.Sequential(*linear_relu_stack)

    def forward(self, x):
        return self.linear_relu_stack(x)


class DeepHedger(BaseHedger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        modules = [NeuralNetwork(n_input=2*self.n_dim, n_output=self.n_dim, n_layers=3, n_nodes=self.n_dim+15) for _ in range(self.n_steps)]
        self.neural_networks = nn.ModuleList(modules=modules)

    def delta(self, step, spot, prev_hedge, **kwargs):
        input = torch.cat((spot.log(), prev_hedge), dim=1)
        return self.neural_networks[step](input)

    def price(self, derivative, risk_measure, n_samples=100000):
        x = self.model.generate(n_samples, self.n_dim, self.n_steps)
        self.eval()
        with torch.inference_mode(True):
            hedge, spots, costs = self(x)
            PnL = self.compute_pnl(hedge, spots, costs, derivative=derivative, price=None)
            price = risk_measure.compute_risk(PnL)
        return price


class SimpleDeepHedger(DeepHedger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        modules = [NeuralNetwork(n_input=self.n_dim, n_output=self.n_dim, n_layers=3, n_nodes=self.n_dim+15) for _ in range(self.n_steps)]
        self.neural_networks = nn.ModuleList(modules=modules)

    def delta(self, step, spot, **kwargs):
        input = spot.log()
        return self.neural_networks[step](input)


def train(deep_hedger, 
          derivative,
          risk_measure,
          optimizer,
          price=None,
          n_samples=10000,
          validation_samples=None,
          n_epochs=10, 
          batch_size=256, 
          verbose=True):

    n_dim = deep_hedger.n_dim
    n_steps = deep_hedger.n_steps
    
    # create training and test data
    train_data = deep_hedger.model.generate(n_samples, n_dim, n_steps)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, num_workers=2)
    if validation_samples is None:
        validation_samples = n_samples // 4
    test_data = deep_hedger.model.generate(validation_samples, n_dim, n_steps)
    train_history, test_history = [], []

    epoch_progress = trange(n_epochs)
    for epoch in epoch_progress:
        # Compute training loss and backpropagate
        running_train_loss = 0.0
        batch_progress = tqdm(train_dataloader, unit='batch', leave=False, disable=not verbose)
        for batch in batch_progress:
            deep_hedger.train()
            with torch.set_grad_enabled(True):
                optimizer.zero_grad()
                hedge, spots, costs = deep_hedger(batch)
                PnL = deep_hedger.compute_pnl(hedge, spots, costs, derivative=derivative, price=price)
                train_loss = risk_measure.compute_risk(PnL)
                train_loss.backward()
                #nn.utils.clip_grad_norm_(deep_hedger.parameters(), max_norm=2.0)
                optimizer.step()
            running_train_loss += train_loss.item()
        train_history.append(running_train_loss / len(batch_progress))

        # Compute validation loss
        deep_hedger.eval()
        with torch.inference_mode(True):
            hedge, spots, costs = deep_hedger(test_data)
            PnL = deep_hedger.compute_pnl(hedge, spots, costs, derivative=derivative, price=price)
            test_loss = risk_measure.compute_risk(PnL)

        test_history.append(test_loss.item())
        epoch_progress.desc = "Training Loss={:.4f} | Validation Loss={:.4f}".format(train_history[-1], test_history[-1])

    return train_history, test_history