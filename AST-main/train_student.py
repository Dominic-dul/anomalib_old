import numpy as np
import torch
from os.path import join
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import config as c
from model import *
from utils import *

# Function to train the student model.
def train(train_loader, test_loader):
    # Initialize the student model with specific configuration for the asymmetric approach.
    student = Model(nf=not c.asymmetric_student, channels_hidden=c.channels_hidden_student, n_blocks=c.n_st_blocks)
    # Move the student model to the configured device (CPU/GPU).
    student.to(c.device)

    # Load the pre-trained teacher model.
    teacher = Model()
    teacher.net.load_state_dict(torch.load(os.path.join(MODEL_DIR, c.modelname + '_' + c.class_name + '_teacher.pth')))
    # Set the teacher model to evaluation mode.
    teacher.eval()
    # Move the teacher model to the configured device.
    teacher.to(c.device)

    # Define the optimizer for the student model.
    optimizer = torch.optim.Adam(student.net.parameters(), lr=c.lr, eps=1e-08, weight_decay=1e-5)

    # Observers to track the AUROC scores during training.
    max_st_obs = Score_Observer('AUROC  max over maps')
    mean_st_obs = Score_Observer('AUROC mean over maps')

    # Training loop across specified epochs.
    for epoch in range(c.meta_epochs):
        # Set student model to training mode.
        student.train()
        if c.verbose:
            print(F'\nTrain epoch {epoch}')
        # Iterate over batches from the training loader.
        for sub_epoch in range(c.sub_epochs):
            # To accumulate training losses.
            train_loss = list()
            for i, data in enumerate(tqdm(train_loader, disable=c.hide_tqdm_bar)):
                # Move data to the configured device.
                depth, fg, labels, image, features = data
                depth, fg, image, features = to_device([depth, fg, image, features])
                # Optionally apply dilation to the foreground mask.
                fg = dilation(fg, c.dilate_size) if c.dilate_mask else fg

                # Clear gradients before each optimization step.
                optimizer.zero_grad()
                img_in = features if c.pre_extracted else image
                fg_down = downsampling(fg, (c.map_len, c.map_len), bin=False)

                # Get predictions from the teacher model for comparison (no gradient needed).
                with torch.no_grad():
                    z_t, jac_t = teacher(img_in, depth)

                # Get predictions from the student model.
                z, jac = student(img_in, depth)
                # Calculate the loss between the teacher and student predictions.
                loss = get_st_loss(z_t, z, fg_down)
                # Backpropagate errors.
                loss.backward()
                # Update the student model parameters.
                optimizer.step()

                # Store the loss for this batch.
                train_loss.append(t2np(loss))

            # Calculate the mean training loss for the epoch.
            mean_train_loss = np.mean(train_loss)
            if c.verbose and sub_epoch % 4 == 0:  # and epoch == 0:
                print('Epoch: {:d}.{:d} \t train loss: {:.4f}'.format(epoch, sub_epoch, mean_train_loss))

        # Set student model to evaluation mode.
        student.eval()
        if c.verbose:
            print('\nCompute loss and scores on test set:')
        test_loss = list()
        test_labels = list()
        mean_st = list()
        max_st = list()

        # Disable gradient computation for evaluation.
        with torch.no_grad():
            for i, data in enumerate(tqdm(test_loader, disable=c.hide_tqdm_bar)):
                # Move data to the configured device.
                depth, fg, labels, image, features = data
                depth, fg, image, features = to_device([depth, fg, image, features])
                fg = dilation(fg, c.dilate_size) if c.dilate_mask else fg

                img_in = features if c.pre_extracted else image
                fg_down = downsampling(fg, (c.map_len, c.map_len), bin=False)
                # Teacher predictions for comparison.
                z_t, jac_t = teacher(img_in, depth)

                # Student predictions.
                z, jac = student(img_in, depth)

                # Calculate loss for the student based on its difference from the teacher.
                st_loss = get_st_loss(z_t, z, fg_down, per_sample=True)
                # Per-pixel loss for detailed evaluation.
                st_pixel = get_st_loss(z_t, z, fg_down, per_pixel=True)

                # Apply evaluation mask if configured.
                if c.eval_mask:
                    st_pixel = st_pixel * fg_down[:, 0]

                # Store mean loss for evaluation.
                mean_st.append(t2np(st_loss))
                # Store max loss for evaluation.
                max_st.append(np.max(t2np(st_pixel), axis=(1, 2)))
                # Accumulate test loss.
                test_loss.append(st_loss.mean().item())
                # Store labels for AUROC calculation.
                test_labels.append(labels)

        # Flatten list of mean student losses.
        mean_st = np.concatenate(mean_st)
        # Flatten list of max student losses.
        max_st = np.concatenate(max_st)
        # Calculate mean test loss.
        test_loss = np.mean(np.array(test_loss))

        if c.verbose:
            print('Epoch: {:d} \t test_loss: {:.4f}'.format(epoch, test_loss))

        # Flatten list of test labels.
        test_labels = np.concatenate(test_labels)
        # Convert labels to binary anomaly indicator.
        is_anomaly = np.array([0 if l == 0 else 1 for l in test_labels])

        # Update AUROC observers with the new scores and print if verbose.
        mean_st_obs.update(roc_auc_score(is_anomaly, mean_st), epoch,
                           print_score=c.verbose or epoch == c.meta_epochs - 1)
        max_st_obs.update(roc_auc_score(is_anomaly, max_st), epoch, print_score=c.verbose or epoch == c.meta_epochs - 1)

    # Optionally save the trained student model's weights.
    if c.save_model:
        save_weights(student, 'student')

    return mean_st_obs, max_st_obs


if __name__ == "__main__":
    train_dataset(train)
