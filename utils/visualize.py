import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE


def visualize(image_data, label="no label"):
    if isinstance(image_data, torch.Tensor):
        image_data = image_data.detach().cpu().numpy()

    image_data = np.transpose(image_data, (1, 2, 0))

    image_data = np.clip(image_data, 0, 1)

    plt.imshow(image_data)
    plt.axis('off')
    plt.text(0, -2, label, color='black')
    plt.show()


def visualize_batch(image_data_batch, label_batch=None):
    len_data_batch = len(image_data_batch)
    rows = int(np.sqrt(len_data_batch))
    cols = int(np.ceil(len_data_batch / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    if label_batch is None:
        label_batch = [f"None" for i in range(len_data_batch)]

    if isinstance(image_data_batch, torch.Tensor):
        image_data_batch = image_data_batch.detach().cpu().numpy()

    for i, ax in enumerate(axes.flat):
        try:
            # Display image
            sample = image_data_batch[i - 1]
            image_data = np.transpose(sample, (1, 2, 0))
            ax.imshow(image_data)
            ax.set_title(f"Label: {label_batch[i - 1]}")
            ax.axis('off')
        except:
            continue
    plt.tight_layout()
    plt.show()


def visualize_tsne(features, labels_tensor, attaches=None, save_path=None):
    if isinstance(features, torch.Tensor):
        features_np = features.cpu().detach().numpy()
        labels_np = labels_tensor.cpu().detach().numpy()
    else:
        features_np = features
        labels_np = labels_tensor

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=1234)
    features_tsne = tsne.fit_transform(features_np)

    # Plot the t-SNE visualization
    plt.figure(figsize=(10, 10))
    color_list = ['#003366', '#2F6027', '#801E23']
    clean_color_list = ['#ea5545', '#f46a9b', '#ef9b20', '#edbf33', '#27aeef',
                        '#b33dc6', '#87bc45', '#00bfa0', '#9b19f5', '#ffa300']

    for label in np.unique(labels_np):
        if label >= 100:
            plt.scatter(features_tsne[labels_np == label, 0], features_tsne[labels_np == label, 1],
                        label=str(label), edgecolors='w', linewidths=0.5, color=color_list[int(label - 100)], alpha=0.4)
        else:
            plt.scatter(features_tsne[labels_np == label, 0], features_tsne[labels_np == label, 1],
                        label=str(label), edgecolors='w', linewidths=0.5, color = clean_color_list[int(label)])
    plt.legend()
    # plt.title('t-SNE visualization')
    plt.show()

    if save_path is not None:
        plt.savefig(f'./{save_path}/tsne_{attaches}.png')
