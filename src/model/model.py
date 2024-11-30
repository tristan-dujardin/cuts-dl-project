import os
import random
import torch
import torch.nn.functional as func
import numpy as np

from skimage.metrics import structural_similarity
from .output_saver import OutputSaver
from tqdm import tqdm

class CUTSModel(torch.nn.Module):

    def __init__(self, input_channels, kernels=16, patch_size=5, sampled_patches_per_image=4):
        super(CUTSModel, self).__init__()

        self.latent_dim = kernels * 8
        self.patch_size = patch_size
        self.recon = Patch(input_channels, patch_size, self.latent_dim)
        self.patch_sampler = PatchSampler(
            random_seed=42,
            patch_size=self.patch_size,
            sampled_patches_per_image=sampled_patches_per_image)


        self.batch_norm1 = torch.nn.BatchNorm2d(kernels)
        self.batch_norm2 = torch.nn.BatchNorm2d(kernels * 2)
        self.batch_norm3 = torch.nn.BatchNorm2d(kernels * 4)
        self.batch_norm4 = torch.nn.BatchNorm2d(kernels * 8)

        self.conv1 = torch.nn.Conv2d(input_channels, kernels, kernel_size=5, padding='same', padding_mode='replicate')
        self.conv2 = torch.nn.Conv2d(kernels, kernels*2, kernel_size=5, padding='same', padding_mode='replicate')
        self.conv3 = torch.nn.Conv2d(kernels*2, kernels*4, kernel_size=5, padding='same', padding_mode='replicate')
        self.conv4 = torch.nn.Conv2d(kernels*4, kernels*8, kernel_size=5, padding='same', padding_mode='replicate')

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
                patch_real[batch_index, sample_index, ...] = x[batch_index, :, anchors_hw[batch_index, sample_index, 0]
                                                               - self.patch_size // 2:anchors_hw[batch_index, sample_index, 0]
                                                               - self.patch_size // 2 + self.patch_size, anchors_hw[batch_index, sample_index, 1]
                                                               - self.patch_size // 2:anchors_hw[batch_index, sample_index, 1]
                                                               - self.patch_size // 2 + self.patch_size]
                tmp_1 = anchors_hw[batch_index, sample_index, 0]
                tmp_2 = anchors_hw[batch_index, sample_index, 1]
                W_anchors[batch_index, sample_index, ...] = W[batch_index, :, tmp_1, tmp_2]
                W_positives[batch_index, sample_index, ...] = W[batch_index, :, positives_hw[batch_index, sample_index, 0], positives_hw[batch_index, sample_index, 1]]

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
    def __init__(self, input_channels, patch_size, latent_dim):
        super(Patch, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.recon = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, self.input_channels * self.patch_size**2),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.input_channels * self.patch_size**2, self.input_channels * self.patch_size**2)
        )

    def forward(self, x):
        B, S, _ = x.shape
        C = self.input_channels
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
    

class PatchSampler(object):
    def __init__(self, random_seed=None, patch_size=None, sampled_patches_per_image=None):
        self.random_seed = random_seed
        self.patch_size = patch_size
        self.sampled_patches_per_image = sampled_patches_per_image
        self.max_attempts = 20
        self.ssim_thr = 0.5

    def sample(self, image):
        B, _, H, W = image.shape

        anchors_hw = np.zeros((B, self.sampled_patches_per_image, 2),
                              dtype=int)
        positives_hw = np.zeros_like(anchors_hw)

        h_range = (self.patch_size // 2, H - self.patch_size // 2)
        w_range = (self.patch_size // 2, W - self.patch_size // 2)

        random.seed(self.random_seed)
        for batch_idx in range(B):
            for sample_idx in range(self.sampled_patches_per_image):
                anchors_hw[batch_idx, sample_idx, :] = [
                    random.randrange(start=h_range[0], stop=h_range[1]),
                    random.randrange(start=w_range[0], stop=w_range[1]),
                ]
                best_pos_hw_candidate = None
                for _ in range(self.max_attempts):
                    pos_hw_candidate = sample_hw_nearby(
                        anchors_hw[batch_idx, sample_idx, :],
                        H=H,
                        W=W,
                        neighborhood=self.patch_size,
                        patch_size=self.patch_size)

                    curr_ssim = compute_ssim(image[batch_idx, ...].cpu().detach().numpy(),
                                             h1w1=anchors_hw[batch_idx,
                                                             sample_idx, :],
                                             h2w2=pos_hw_candidate,
                                             patch_size=self.patch_size)

                    if curr_ssim > self.ssim_thr:
                        best_pos_hw_candidate = pos_hw_candidate
                        break

                if best_pos_hw_candidate is None:
                    neighbor_hw = anchors_hw[batch_idx, sample_idx, :]
                    neighbor_hw[0] = neighbor_hw[0] - 1 if neighbor_hw[
                        0] > H // 2 else neighbor_hw[0] + 1
                    neighbor_hw[1] = neighbor_hw[1] - 1 if neighbor_hw[
                        1] > W // 2 else neighbor_hw[1] + 1
                    best_pos_hw_candidate = neighbor_hw

                positives_hw[batch_idx, sample_idx, :] = best_pos_hw_candidate

        assert anchors_hw.shape == positives_hw.shape
        return anchors_hw, positives_hw


def sample_hw_nearby(hw, H, W, neighborhood=5, patch_size=7):
    h_start = max(hw[0] - neighborhood, patch_size // 2)
    h_stop = min(hw[0] + neighborhood, H - patch_size // 2)
    w_start = max(hw[1] - neighborhood, patch_size // 2)
    w_stop = min(hw[1] + neighborhood, W - patch_size // 2)

    return (random.randrange(start=h_start, stop=h_stop),
            random.randrange(start=w_start, stop=w_stop))

def ssim(a, b, **kwargs):
    assert a.shape == b.shape

    H, W = a.shape[:2]

    if min(H, W) < 7:
        win_size = min(H, W)
        if win_size % 2 == 0:
            win_size -= 1
    else:
        win_size = None

    if len(a.shape) == 3:
        channel_axis = -1
    else:
        channel_axis = None

    return structural_similarity(a,
                                 b,
                                 channel_axis=channel_axis,
                                 win_size=win_size,
                                 **kwargs)


def range_aware_ssim(label_true, label_pred):

    if isinstance(label_true.max(), bool):
        label_true = label_true.astype(np.float32)
        label_pred = label_pred.astype(np.float32)
    data_range = label_true.max() - label_true.min()

    if data_range == 0:
        data_range = 1.0

    return ssim(a=label_true, b=label_pred, data_range=data_range)

def compute_ssim(image, h1w1, h2w2, patch_size):
    patch1 = image[:, h1w1[0] - patch_size // 2:h1w1[0] - patch_size // 2 +
                   patch_size, h1w1[1] - patch_size // 2:h1w1[1] -
                   patch_size // 2 + patch_size]
    patch2 = image[:, h2w2[0] - patch_size // 2:h2w2[0] - patch_size // 2 +
                   patch_size, h2w2[1] - patch_size // 2:h2w2[1] -
                   patch_size // 2 + patch_size]

    patch1 = np.moveaxis(patch1, 0, -1)
    patch2 = np.moveaxis(patch2, 0, -1)
    return range_aware_ssim(patch1, patch2)

##############################################################################################################################
### Model Training & Eval
    
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
            print(f'Epoch {epoch:>3} | Val Loss: {val_loss:.2f} | Val Loss Recon: {val_loss_recon*100:>5.2f}% | Val Loss Contrastive: {val_loss_contrastive:.2f}')
        
    return model

@torch.no_grad()
def test(model, test_loader, lambda_contr_loss=0.001):
    output_saver = OutputSaver('images/')

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

            B, L, H, Wi = W.shape
            W_for_recon = W.permute((0, 2, 3, 1)).reshape(B, H * Wi, L)
            patch_recon = model.recon(W_for_recon)
            C = patch_recon.shape[2]
            P = patch_recon.shape[-1]
            patch_recon = patch_recon[:, :, :, P // 2, P // 2]
            patch_recon = patch_recon.permute((0, 2, 1)).reshape(B, C, H, Wi)

            output_saver.save(image_batch=data_x,
                              recon_batch=patch_recon,
                              label_true_batch=data_y,
                              latent_batch=W)
    test_loss_recon = test_loss_recon / len(test_loader.dataset)
    test_loss_contrastive = test_loss_contrastive / len(test_loader.dataset)
    test_loss = test_loss / len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.2f} | Test Loss Recon: {test_loss_recon:.2f}% | Test Loss Contrastive: {test_loss_contrastive:.2f}')

    return test_loss