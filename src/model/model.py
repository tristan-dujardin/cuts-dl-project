import os
import torch
import torch.nn.functional as func

from tqdm import tqdm

class CUTSModel(torch.nn.Module):

    def __init__(self, input_size, kernels, patch_size):
        super(CUTSModel, self).__init__()

        self.patch_size = patch_size
        self.recon = Patch(input_size, patch_size, kernels * 8)

        self.batch_norm1 = torch.nn.BatchNorm2d(kernels)
        self.batch_norm2 = torch.nn.BatchNorm2d(kernels * 2)
        self.batch_norm3 = torch.nn.BatchNorm2d(kernels * 4)
        self.batch_norm4 = torch.nn.BatchNorm2d(kernels * 8)

        self.conv1 = torch.nn.Conv2d(input_size, kernels)
        self.conv2 = torch.nn.Conv2d(kernels, kernels*2)
        self.conv3 = torch.nn.Conv2d(kernels*2, kernels*4)
        self.conv4 = torch.nn.Conv2d(kernels*4, kernels*8)

    def forward(self, x):

        B, C, _, _ = x.shape

        W = func.leaky_relu(self.batch_norm1(self.conv1(x)))
        W = func.leaky_relu(self.batch_norm2(self.conv2(W)))
        W = func.leaky_relu(self.batch_norm3(self.conv3(W)))
        W = func.leaky_relu(self.batch_norm4(self.conv4(W)))

        anchors_hw, positives_hw = self.patch_sampler.sample(x)
        S = anchors_hw.shape[1]
        patch_real = torch.zeros((B, S, C, self.patch_size, self.patch_size))
        W_anchors = torch.zeros((B, S, self.latent_dim))
        W_positives = torch.zeros_like(W_anchors)

        assert anchors_hw.shape[0] == B
        for batch_index in range(B):
            for sample_index in range(S):
                patch_real[batch_index, sample_index, ...] = x[
                    batch_index, :, anchors_hw[batch_index, sample_index, 0] -
                    self.patch_size // 2:anchors_hw[batch_index, sample_index, 0] -
                    self.patch_size // 2 + self.patch_size,
                    anchors_hw[batch_index, sample_index, 1] -
                    self.patch_size // 2:anchors_hw[batch_index, sample_index, 1] -
                    self.patch_size // 2 + self.patch_size]
                W_anchors[batch_index, sample_index,
                              ...] = W[batch_index, :, anchors_hw[batch_index, sample_index, 0],
                                       anchors_hw[batch_index, sample_index, 1]]
                W_positives[batch_index, sample_index,
                                ...] = W[batch_index, :,
                                         positives_hw[batch_index, sample_index,
                                                      0],
                                         positives_hw[batch_index, sample_index,
                                                      1]]

        patch_recon = self.recon(W_anchors)

        return W, patch_real, patch_recon, W_anchors, W_positives
    
    def save_model(self, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        torch.save(self.state_dict(), file_path)
        return

    def load_model(self, file_path):
        self.load_state_dict(torch.load(file_path))
        return

class Patch(torch.nn.Module):
    def __init__(self, input_size, latent_dim, patch_size):
        super(Patch, self).__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.recon = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, self.input_size * self.patch_size**2),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.in_channels * self.patch_size**2, self.input_size * self.patch_size**2)
        )

    def forward(self, x):
        B, S, _ = x.shape
        C = self.in_channels
        P = self.patch_size

        reconed_patch = None

        for batch_index in range(B):
            curr_recon = self.recon(x[batch_index, ...]).reshape(S, C, P, P)
            if reconed_patch is None:
                reconed_patch = curr_recon.unsqueeze(0)
            else:
                reconed_patch = torch.cat(
                    (reconed_patch, curr_recon.unsqueeze(0)), dim=0)

        return reconed_patch


def train(model, train_loader, val_loader, params, verbose=False):
    def __acc(pred_y, y): return ((pred_y == y).sum() / len(y)).item()

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)

    model.train()
    for epoch in tqdm(range(params.epochs)):
        train_loss_recon, train_loss_contrastive, train_loss = 0, 0, 0
        for data in train_loader:
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