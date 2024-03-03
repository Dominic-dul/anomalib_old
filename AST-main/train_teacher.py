import numpy as np
import torch
from os.path import join
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import config as c
from model import *
from utils import *


def train(train_loader, test_loader):
    model = Model()
    # Move model to configured device (GPU/CPU).
    model.to(c.device)
    # Set up optimizer with specified learning rate and weight decay.
    optimizer = torch.optim.Adam(model.net.parameters(), lr=c.lr, eps=1e-08, weight_decay=1e-5)

    # Observers to track AUROC scores during training.
    mean_nll_obs = Score_Observer('AUROC mean over maps')
    max_nll_obs = Score_Observer('AUROC  max over maps')

    # Main training loop across epochs.
    for epoch in range(c.meta_epochs):
        # Set model to training mode.
        model.train()
        if c.verbose:
            print(F'\nTrain epoch {epoch}')
        for sub_epoch in range(c.sub_epochs):
            # To track training loss.
            train_loss = list()
            for i, data in enumerate(tqdm(train_loader, disable=c.hide_tqdm_bar)):
                # Clear gradients.
                optimizer.zero_grad()

                # Unpack data and move to device.
                depth, fg, labels, image, features = data
                depth, fg, labels, image, features = to_device([depth, fg, labels, image, features])
                # Optionally apply dilation to the foreground mask.
                fg = dilation(fg, c.dilate_size) if c.dilate_mask else fg

                # Choose input based on whether features are pre-extracted.
                img_in = features if c.pre_extracted else image
                # Downsample foreground mask to match the model output size.
                fg_down = downsampling(fg, (c.map_len, c.map_len), bin=False)
                # Forward pass through the model.
                z, jac = model(img_in, depth)

                # Calculate loss and backpropagate.
                loss = get_nf_loss(z, jac, fg_down)
                # Convert tensor loss to numpy and store.
                train_loss.append(t2np(loss))

                # Compute gradients.
                loss.backward()
                # Update model parameters.
                optimizer.step()

            # Calculate mean training loss for the epoch.
            mean_train_loss = np.mean(train_loss)
            if c.verbose and sub_epoch % 4 == 0:  # and epoch == 0:
                print('Epoch: {:d}.{:d} \t train loss: {:.4f}'.format(epoch, sub_epoch, mean_train_loss))

        # Evaluation phase.
        model.eval()
        if c.verbose:
            print('\nCompute loss and scores on test set:')
         # To track test loss.
        test_loss = list()
        # To collect labels for AUROC computation.
        test_labels = list()
        # To track per-image loss.
        img_nll = list()
        # To track max loss over maps.
        max_nlls = list()
        # Disable gradient computation for evaluation.
        with torch.no_grad():
            for i, data in enumerate(tqdm(test_loader, disable=c.hide_tqdm_bar)):
                # Unpack and move data to device, similar to training phase.
                depth, fg, labels, image, features = data
                depth, fg, image, features = to_device([depth, fg, image, features])

                fg = dilation(fg, c.dilate_size) if c.dilate_mask else fg

                img_in = features if c.pre_extracted else image
                fg_down = downsampling(fg, (c.map_len, c.map_len), bin=False)
                z, jac = model(img_in, depth)
                # Calculate loss for each sample.
                loss = get_nf_loss(z, jac, fg_down, per_sample=True)
                # Calculate per-pixel loss.
                nll = get_nf_loss(z, jac, fg_down, per_pixel=True)

                img_nll.append(t2np(loss))
                # Track max loss over all pixels.
                max_nlls.append(np.max(t2np(nll), axis=(-1, -2)))
                # Calculate mean test loss.
                test_loss.append(loss.mean().item())
                test_labels.append(labels)

        # Prepare loss and label data for AUROC computation.
        img_nll = np.concatenate(img_nll)
        max_nlls = np.concatenate(max_nlls)
        test_loss = np.mean(np.array(test_loss))

        if c.verbose:
            print('Epoch: {:d} \t test_loss: {:.4f}'.format(epoch, test_loss))

        test_labels = np.concatenate(test_labels)
        # Prepare anomaly labels.
        is_anomaly = np.array([0 if l == 0 else 1 for l in test_labels])

        # Update AUROC observers and print scores.
        mean_nll_obs.update(roc_auc_score(is_anomaly, img_nll), epoch,
                            print_score=c.verbose or epoch == c.meta_epochs - 1)
        max_nll_obs.update(roc_auc_score(is_anomaly, max_nlls), epoch,
                           print_score=c.verbose or epoch == c.meta_epochs - 1)

    # Optionally save the trained teacher model.
    if c.save_model:
        save_weights(model, 'teacher')

    return mean_nll_obs, max_nll_obs


if __name__ == "__main__":
    train_dataset(train)
