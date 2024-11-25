import os
import torch


class CUTSModel(torch.nn.Module):
    def __init__(self):
        super(CUTSModel, self).__init__()
    def forward(self, x):
        return x
    
    def save_model(self, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        torch.save(self.state_dict(), file_path)
        return

    def load_model(self, file_path):
        self.load_state_dict(torch.load(file_path))
        return


def train(model, train_loader, val_loader, params, verbose=False):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

    model.train()
    # TODO

    return model

@torch.no_grad()
def test(model, test_loader):
    criterion = torch.nn.MSELoss()
    model.eval()
    # TODO
    
    return 1