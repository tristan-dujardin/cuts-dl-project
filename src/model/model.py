import os
import torch
import torch.nn as nn
import torch.nn.functional as func

from tqdm import tqdm

class CUTSModel(torch.nn.Module):

    def __init__(self, input_size, kernels):
        super(CUTSModel, self).__init__()

        self.conv1 = nn.Conv2d(input_size, kernels)
        self.conv2 = nn.Conv2d(kernels, kernels*2)
        self.conv3 = nn.Conv2d(kernels*2, kernels*4)
        self.conv4 = nn.Conv2d(kernels*4, kernels*8)

    def forward(self, x):
        W = func.leaky_relu(self.conv1(x))
        W = func.leaky_relu(self.conv2(W))
        W = func.leaky_relu(self.conv3(W))
        W = func.leaky_relu(self.conv4(W))

        return W
    
    def save_model(self, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        torch.save(self.state_dict(), file_path)
        return

    def load_model(self, file_path):
        self.load_state_dict(torch.load(file_path))
        return


def train(model, train_loader, val_loader, params, verbose=False):
    def __acc(pred_y, y): return ((pred_y == y).sum() / len(y)).item()

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)

    model.train()
    for epoch in tqdm(range(params.epochs)):
        for data in train_loader:
            """J'ai pas regarde le dataset mais la c'est un placeholder de donnees labellisees
            mtn il fait aller regarder le train de leur implementation et faire un truc similaire"""
            #TODO
            optimizer.zero_grad()

            out = model(data.x)
            loss = criterion(out, data.y)
            total_loss += loss / len(train_loader)
            acc += __acc(out.argmax(dim=1), data.y) / len(train_loader) 
            loss.backward()
            optimizer.step()


            if(verbose and epoch % 10 == 0):
                val_loss, val_acc = test(model, val_loader)
                print(f'Epoch {epoch:>3} | Train Loss: {total_loss:.2f} | Train Acc: {acc*100:>5.2f}% | Val Loss: {val_loss:.2f} | Val Acc: {val_acc*100:.2f}%')
        
    return model

@torch.no_grad()
def test(model, test_loader):
    criterion = torch.nn.MSELoss()
    model.eval()
    # TODO

    return 1