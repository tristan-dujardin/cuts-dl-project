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
    
class NTXentLoss(torch.nn.Module):

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, anchors, positives):
        assert len(anchors.shape) == 3
        assert anchors.shape == positives.shape

        B, S, _ = anchors.shape

        loss = 0
        for batch_idx in range(B):
            Z_anchors = anchors[batch_idx, ...]
            Z_pos = positives[batch_idx, ...]

            assert Z_anchors.shape == Z_pos.shape

            z1 = torch.nn.functional.normalize(Z_anchors, p=2, dim=1)
            z2 = torch.nn.functional.normalize(Z_pos, p=2, dim=1)
            z = torch.cat((z1, z2), dim=0)

            sim_matrix = torch.exp(torch.matmul(z, z.T) / self.temperature)

            positive_mask = torch.cat((
                torch.cat((torch.zeros((S, S), dtype=torch.bool), torch.eye(S, dtype=torch.bool)), dim=0),
                torch.cat((torch.eye(S, dtype=torch.bool), torch.zeros((S, S), dtype=torch.bool)), dim=0),
                                    ), dim=1)
            negative_mask = torch.cat((
                torch.cat((~torch.eye(S, dtype=torch.bool), ~torch.eye(S, dtype=torch.bool)), dim=0),
                torch.cat((~torch.eye(S, dtype=torch.bool), ~torch.eye(S, dtype=torch.bool)), dim=0),
                                    ), dim=1)

            score_pos = sim_matrix[positive_mask].view(2 * S, 1)

            score_neg = sim_matrix[negative_mask].view(2 * S, -1).sum(dim=1, keepdim=True)

            loss += -torch.log(score_pos / (score_pos + score_neg))

        return loss.mean() / B


def train(model, train_loader, val_loader, params, lambda_contr_loss=0.001, verbose=False):
    def __acc(pred_y, y): return ((pred_y == y).sum() / len(y)).item()

    criterion = torch.nn.MSELoss()
    criterion_contrastive = NTXentLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)

    model.train()
    for epoch in tqdm(range(params.epochs)):
        train_loss_recon, train_loss_contrastive, train_loss = 0, 0, 0

        for _, (data, _) in enumerate(train_loader):
            optimizer.zero_grad()

            B = data.shape[0]
            data = data.type(torch.FloatTensor)

            _, patch_real, patch_recon, z_anchors, z_positives = model(data)

            loss_recon = criterion(patch_real, patch_recon)
            loss_contrastive = criterion_contrastive(z_anchors, z_positives)
            loss = lambda_contr_loss * loss_contrastive + (1 - lambda_contr_loss) * loss_recon
            
            loss.backward()
            optimizer.step()

            train_loss_recon += loss_recon.item() * B
            train_loss_contrastive += loss_contrastive.item() * B
            train_loss += loss.item() * B
        
        train_loss_recon /= len(train_loader.dataset)
        train_loss_contrastive /= len(train_loader.dataset)
        train_loss /= len(train_loader.dataset)
        
        if(verbose and epoch % 10 == 0):
            val_loss_recon, val_loss_contrastive, val_loss = 0, 0, 0
            model.eval()
            with torch.no_grad():
                for _, (x_val, _) in enumerate(val_loader):
                    B = x_val.shape[0]

                    x_val = x_val.type(torch.FloatTensor)
                    _, patch_real, patch_recon, z_anchors, z_positives = model(
                        x_val)

                    loss_recon = criterion(patch_real, patch_recon)
                    loss_contrastive = criterion_contrastive(z_anchors, z_positives)
                    loss = lambda_contr_loss * loss_contrastive + (1 - lambda_contr_loss) * loss_recon

                    val_loss_recon += loss_recon.item() * B
                    val_loss_contrastive += loss_contrastive.item() * B
                    val_loss += loss.item() * B

                val_loss_recon /= len(val_loader.dataset)
                val_loss_contrastive /= len(val_loader.dataset)
                val_loss /= len(val_loader.dataset)
            print(f'Epoch {epoch:>3} | Val Loss: {val_loss:.2f} | Val Loss Contrastive: {val_loss_contrastive*100:>5.2f}% | Val Loss Contrastive: {val_loss_recon:.2f}')
        
    return model

@torch.no_grad()
def test(model, test_loader, lambda_contr_loss=0.001):
    criterion = torch.nn.MSELoss()
    criterion_contrastive = NTXentLoss()
    test_loss_recon, test_loss_contrastive, test_loss = 0, 0, 0

    model.eval()
    with torch.no_grad():
        for _, (data_x, data_y) in enumerate(test_loader):
            B = data_x.shape[0]

            data_x = data_x.type(torch.FloatTensor)
            W, patch_real, patch_recon, z_anchors, z_positives = model(data_x)

            loss_recon = criterion(patch_real, patch_recon)
            loss_contrastive = criterion_contrastive(z_anchors, z_positives)
            loss = lambda_contr_loss * loss_contrastive + (1 - lambda_contr_loss) * loss_recon

            test_loss_recon += loss_recon.item() * B
            test_loss_contrastive += loss_contrastive.item() * B
            test_loss += loss.item() * B

            B, L, H, W = W.shape
            W_for_recon = W.permute((0, 2, 3, 1)).reshape(B, H * W, L)
            patch_recon = model.recon(W_for_recon)
            C = patch_recon.shape[2]
            P = patch_recon.shape[-1]
            patch_recon = patch_recon[:, :, :, P // 2, P // 2]
            patch_recon = patch_recon.permute((0, 2, 1)).reshape(B, C, H, W)

            output_saver.save(image_batch=data_x,
                              recon_batch=patch_recon,
                              label_true_batch=data_y if config.no_label is False else None,
                              latent_batch=W)
    test_loss_recon = test_loss_recon / len(test_loader.dataset)
    test_loss_contrastive = test_loss_contrastive / len(test_loader.dataset)
    test_loss = test_loss / len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.2f} | Test Loss Contrastive: {test_loss_contrastive*100:>5.2f}% | Test Loss Contrastive: {test_loss_recon:.2f}')

    return test_loss