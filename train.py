import jax
import jax.numpy as jnp
import optax
from vit_jax_flax.vit import ViT
from jax import random
import flax
from flax.training import train_state, checkpoints

import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from collections import defaultdict
from tqdm.auto import tqdm


DATA_MEANS = np.array([0.49139968, 0.48215841, 0.44653091])
DATA_STD = np.array([0.24703223, 0.24348513, 0.26158784])


def image_to_numpy(img):
    img = np.array(img, dtype=np.float32)
    img = (img / 255. - DATA_MEANS) / DATA_STD
    return img    


# We need to stack the batch elements
def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

def calculate_loss(params, rng, batch, train):
    imgs, labels = batch
    rng, drop_rng = random.split(rng)
    logits = model.apply({'params': params}, imgs, train=train, rngs={'dropout': drop_rng})
    loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels).mean()
    acc = (logits.argmax(axis=-1) == labels).mean()
    return loss, (acc, rng)


@jax.jit
def train_step(state, rng, batch):
    loss_fn = lambda params: calculate_loss(params, rng, batch, train=True)
    # Get loss, gradients for loss, and other outputs of loss function
    (loss, (acc, rng)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    # Update parameters and batch statistics
    state = state.apply_gradients(grads=grads)
    return state, rng, loss, acc


@jax.jit
def eval_step(state, rng, batch):
    _, (acc, rng) = calculate_loss(state.params, rng, batch, train=False)
    return rng, acc

def train_epoch(train_loader, epoch_idx, state, rng):
    metrics = defaultdict(list)
    for batch in tqdm(train_loader, desc='Training', leave=False):
        state, rng, loss, acc = train_step(state, rng, batch)
        metrics['loss'].append(loss)
        metrics['acc'].append(acc)
    for key in metrics.keys():
        arg_val = np.stack(jax.device_get(metrics[key])).mean()
        logger.add_scalar('train/' + key, arg_val, global_step=epoch_idx)
        print(f'[epoch {epoch_idx}] {key}: {arg_val}')
    return state, rng


def eval_model(val_loader, state, rng):
    # Test model on all images of a data loader and return avg loss
    correct_class, count = 0, 0
    for batch in val_loader:
        rng, acc = eval_step(state, rng, batch)
        correct_class += acc * batch[0].shape[0]
        count += batch[0].shape[0]
    eval_acc = (correct_class / count).item()
    return eval_acc


def save_model(state, step=0):
    checkpoints.save_checkpoint(ckpt_dir='vit_jax_logs', target=state.params, step=step, overwrite=True)


def train_model(train_loader, val_loader, state, rng, num_epochs=200):
    best_eval = 0.0
    for epoch_idx in tqdm(range(1, num_epochs + 1)):
        state, rng = train_epoch(train_loader, epoch_idx, state, rng)
    if epoch_idx % 1 == 0:
        eval_acc = eval_model(val_loader, state, rng)
        logger.add_scalar('val/acc', eval_acc, global_step=epoch_idx)
        if eval_acc >= best_eval:
            best_eval = eval_acc
            save_model(state, step=epoch_idx)
        logger.flush()
    # Evaluate after training
    print(eval_model(val_loader, state, rng))

def init_train_state(model, params, learning_rate):
    optimizer = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        tx=optimizer,
        params=params
    )


if __name__ == '__main__':
    main_rng = jax.random.PRNGKey(42)
    x = jnp.ones(shape=(5, 32, 32, 3))

    # ViT
    model = ViT(
        patch_size=4,
        embed_dim=256,
        hidden_dim=512,
        n_heads=8,
        drop_p=0.2,
        num_layers=6,
        mlp_dim=1024,
        num_classes=10
    )
    main_rng, init_rng, drop_rng = random.split(main_rng, 3)
    params = model.init({'params': init_rng, 'dropout': drop_rng}, x, train=True)['params']

    # Dataset preparation
    test_transform = image_to_numpy
    # For training, we add some augmentations. Neworks are too powerful and would overfit.
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        image_to_numpy
    ])

    train_dataset = CIFAR10('data', train=True, transform=train_transform, download=True)
    val_dataset = CIFAR10('data', train=True, transform=test_transform, download=True)
    train_set, _ = torch.utils.data.random_split(train_dataset, [45000, 5000], generator=torch.Generator().manual_seed(42))
    _, val_set = torch.utils.data.random_split(val_dataset, [45000, 5000], generator=torch.Generator().manual_seed(42))
    test_set = CIFAR10('data', train=False, transform=test_transform, download=True)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=128, shuffle=True, drop_last=True, collate_fn=numpy_collate, num_workers=8, persistent_workers=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=128, shuffle=False, drop_last=False, collate_fn=numpy_collate, num_workers=4, persistent_workers=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=128, shuffle=False, drop_last=False, collate_fn=numpy_collate, num_workers=4, persistent_workers=True
    )

    # Training ViT
    logger = SummaryWriter(log_dir='vit_jax_logs')
    state = init_train_state(model, params, 3e-4)
    train_model(train_loader, val_loader, state, main_rng, num_epochs=10)
