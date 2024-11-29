import os
import phate
import scprep
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
from sklearn.preprocessing import normalize

def continuous_renumber(label_orig):
    label = label_orig.copy()
    val_before = np.unique(label_orig)
    val_after = np.arange(len(val_before))
    for (a, b) in zip(val_before, val_after):
        label[label_orig == a] = b

    return label

def label_hint_seg(label_pred, label_true):
    foreground_xys = np.argwhere(label_true)
    if len(foreground_xys) > 0:
        label_counts = {}
        for (x, y) in foreground_xys:
            label_id = label_pred[x, y]
            if label_id not in label_counts.keys():
                label_counts[label_id] = 1
            else:
                label_counts[label_id] += 1

        label_id, max_count = 0, 0
        for i in label_counts.keys():
            count = label_counts[i]
            if count > max_count:
                max_count = count
                label_id = i

        seg = label_pred == label_id

    else:
        seg = np.zeros(label_pred.shape)

    return seg

def dice_coeff(label_pred, label_true):
    epsilon = 1e-12
    intersection = np.logical_and(label_pred, label_true).sum()
    dice = (2 * intersection + epsilon) / (label_pred.sum() +
                                           label_true.sum() + epsilon)
    return dice

def per_class_dice_coeff(label_pred, label_true):
    dice_list = []
    for class_id in np.unique(label_true)[1:]:
        dice_list.append(
            dice_coeff(label_pred=label_pred == class_id,
                       label_true=label_true == class_id))
    return np.mean(dice_list)

def phate_clustering(latent, random_seed, num_workers):
    phate_operator = phate.PHATE(n_components=3,
                                 knn=100,
                                 n_landmark=500,
                                 t=2,
                                 verbose=False,
                                 random_state=random_seed,
                                 n_jobs=num_workers)

    phate_operator.fit_transform(latent)
    clusters = phate.cluster.kmeans(phate_operator,
                                    n_clusters=10,
                                    random_state=random_seed)
    return clusters

def generate_kmeans(shape, latent, label_true, num_workers=1, random_seed=42):
    H, W, C = shape

    seg_true = label_true > 0

    clusters = phate_clustering(latent, random_seed, num_workers)

    label_pred = clusters.reshape((H, W))

    seg_pred = label_hint_seg(label_pred=label_pred, label_true=label_true)

    return per_class_dice_coeff(seg_pred, seg_true), label_pred, seg_pred

def images_transform(folder_path):
    files_folder = '%s/%s' % (folder_path, 'numpy_files')
    np_files_path = sorted(glob('%s/%s' % (files_folder, '*.npz')))
    save_path_numpy = '%s/%s' % (folder_path, 'numpy_files_seg_kmeans')
    figure_folder = '%s/%s' % (folder_path, 'figures')
    phate_folder = '%s/%s' % (folder_path, 'numpy_files_phate')

    os.makedirs(save_path_numpy, exist_ok=True)
    os.makedirs(figure_folder, exist_ok=True)
    os.makedirs(phate_folder, exist_ok=True)

    dice_list = []
    for image_index in tqdm(range(len(np_files_path))):

        image_path = np_files_path[image_index]
        save_path = '%s/%s' % (save_path_numpy,
                               os.path.basename(np_files_path[image_index]))
        numpy_array = np.load(image_path)
        image = numpy_array['image']
        label_true = numpy_array['label']
        latent = numpy_array['latent']

        image = (image + 1) / 2

        H, W = label_true.shape[:2]
        C = latent.shape[-1]
        X = latent

        dice_score, label_pred, seg_pred = generate_kmeans((H, W, C), latent, label_true)

        with open(save_path, 'wb+') as f:
            np.savez(f,
                         image=image,
                         label=label_true,
                         latent=latent,
                         label_kmeans=label_pred,
                         seg_kmeans=seg_pred)

            print('SUCCESS! %s, dice: %s' % (image_path.split('/')[-1], dice_score))
            dice_list.append(dice_score)

        phate_path = '%s/sample_%s.npz' % (phate_folder,
                                           str(image_index).zfill(5))
        
        phate_op = phate.PHATE(random_state=42)

        data_phate = phate_op.fit_transform(normalize(latent, axis=1))
        with open(phate_path, 'wb+') as f:
            np.savez(f, data_phate=data_phate)

        fig1 = plt.figure(figsize=(15, 4))
        ax = fig1.add_subplot(1, 3, 1)
        # ground truth.
        scprep.plot.scatter2d(data_phate,
                              c=continuous_renumber(
                                  label_true.reshape((H * W, -1))),
                              legend_anchor=(1, 1),
                              ax=ax,
                              title='Ground truth label',
                              xticks=False,
                              yticks=False,
                              label_prefix="PHATE",
                              fontsize=10,
                              s=3)
        ax = fig1.add_subplot(1, 3, 2)
        # kmeans.
        scprep.plot.scatter2d(data_phate,
                              c=continuous_renumber(
                                  label_pred.reshape((H * W, -1))),
                              legend_anchor=(1, 1),
                              ax=ax,    
                              title='Spectral K-means',
                              xticks=False,
                              yticks=False,
                              label_prefix="PHATE",
                              fontsize=10,
                              s=3)
        ax = fig1.add_subplot(1, 3, 3)
        # segmented kmeans.
        scprep.plot.scatter2d(data_phate,
                              c=seg_pred.reshape((H * W, -1)),
                              legend_anchor=(1, 1),
                              ax=ax,
                              title='Spectral K-means',
                              xticks=False,
                              yticks=False,
                              label_prefix="PHATE",
                              fontsize=10,
                              s=3)

        # 2. Segmentation plot.
        fig2 = plt.figure(figsize=(20, 6))
        ax = fig2.add_subplot(1, 4, 1)
        ax.imshow(image)
        ax.set_axis_off()
        ax = fig2.add_subplot(1, 4, 2)
        gt_cmap = 'gray' if len(np.unique(label_true)) <= 2 else 'tab20'
        ax.imshow(continuous_renumber(label_true), cmap=gt_cmap)
        ax.set_axis_off()
        ax = fig2.add_subplot(1, 4, 3)
        ax.imshow(seg_pred, cmap='gray')
        ax.set_title('Spectral K-means')
        ax.set_axis_off()
        ax = fig2.add_subplot(1, 4, 4)
        ax.imshow(continuous_renumber(label_pred), cmap='tab20')
        ax.set_title('Spectral K-means')
        ax.set_axis_off()

        fig_path = '%s/sample_%s' % (figure_folder, str(image_index).zfill(5))

        fig1.tight_layout()
        fig1.savefig('%s_phate_kmeans.png' % fig_path)

        fig2.tight_layout()
        fig2.savefig('%s_segmentation_kmeans.png' % fig_path)