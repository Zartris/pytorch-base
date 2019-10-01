from __future__ import division
from __future__ import print_function

import argparse
import copy
import math
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from utils.model_utils import initialize_model, get_data_transform


######################################################################
# Helper Functions
# ----------------
#
# Before we write the code for adjusting the models, lets define a few
# helper functions.
#
# Model Training and Validation Code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The ``train_model`` function handles the training and validation of a
# given model. As input, it takes a PyTorch model, a dictionary of
# dataloaders, a loss function, an optimizer, a specified number of epochs
# to train and validate for, and a boolean flag for when the model is an
# Inception model. The *is_inception* flag is used to accomodate the
# *Inception v3* model, as that architecture uses an auxiliary output and
# the overall model loss respects both the auxiliary output and the final
# output, as described
# `here <https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958>`__.
# The function trains for the specified number of epochs and after each
# epoch runs a full validation step. It also keeps track of the best
# performing model (in terms of validation accuracy), and at the end of
# training returns the best performing model. After each epoch, the
# training and validation accuracies are printed.
#
def train_model(model, dataloaders, criterion, optimizer, scheduler, class_names, loss, device, output_file,
                num_epochs=25,
                current_epoch=0,
                is_inception=False, best_accuracy=0.0, batch_size=8):
    since = time.time()
    log_dir = Path(output_file.parent, "summary")
    if not log_dir.exists():
        log_dir.mkdir(parents=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_optimizer_wts = copy.deepcopy(optimizer.state_dict())
    best_acc = best_accuracy
    last_saved_loss = loss
    last_saved_epoch = current_epoch

    for epoch in range(current_epoch, num_epochs):
        try:
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0
                first = True
                # Iterate over data.
                for index, (inputs, labels) in enumerate(dataloaders[phase]):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # Get model outputs and calculate loss
                        # Special case for inception because in training it has an auxiliary output. In train
                        #   mode we calculate the loss by summing the final output and the auxiliary output
                        #   but in testing we only consider the final output.
                        if is_inception and phase == 'train':
                            # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                            outputs, aux_outputs = model(inputs)
                            loss1 = criterion(outputs, labels)
                            loss2 = criterion(aux_outputs, labels)
                            loss = loss1 + 0.4 * loss2
                        else:
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                        # else:
                        #     # log data to tensorboard
                        #     if epoch % 20 == 19:
                        #         print("Adding new prediction images")
                        #         writer.add_figure('predictions_' + str(index),
                        #                           plot_classes_preds(model, inputs, labels, class_names, batch_size),
                        #                           global_step=epoch)

                        first = False
                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()
                elif epoch % 20 == 19:
                    writer.add_figure('predictions bads',
                                      visualize_model(model, dataloaders, device, class_names, num_images=20),
                                      global_step=epoch)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # Log data to tensorboard:
                writer.add_scalar('Loss/' + str(phase), epoch_loss, epoch)
                writer.add_scalar('Accuracy/' + str(phase), epoch_acc, epoch)
                writer.add_scalar('LR/' + str(phase), optimizer.state_dict().get('param_groups')[0].get('lr'), epoch)

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_optimizer_wts = copy.deepcopy(optimizer.state_dict())
                    last_saved_loss = epoch_loss
                    save_model(epoch, best_model_wts, best_optimizer_wts, last_saved_loss, best_acc,
                               output_file)
                    writer.add_figure('predictions bads best model',
                                      visualize_model(model, dataloaders, device, class_names, num_images=20),
                                      global_step=epoch)
                    last_saved_epoch = epoch
                # elif phase == 'val' and (epoch - last_saved_epoch) % 50 == 49:
                #     print("Saving because of to long inactivity")
                #     save_model(epoch, best_model_wts, best_optimizer_wts, last_saved_loss, best_acc,
                #                output_file)
                #     model.load_state_dict(best_model_wts)
                if phase == 'val':
                    val_acc_history.append(epoch_acc)
        except KeyboardInterrupt:
            print('manually interrupt, try saving model for now...')
            save_model(epoch, best_model_wts, best_optimizer_wts, last_saved_loss, best_acc, output_file)
            print('model saved.')

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def save_model(epoch, model_state_dict, optimizer_state_dict, loss, acc, PATH):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer_state_dict,
        'loss': loss,
        'accuracy': acc
    }, PATH)
    print("New best model saved (loss :", str(loss), ") (accuracy :", str(acc), ")")


def load_model_data(PATH):
    print("=> loading checkpoint '{}' ...".format(PATH))
    if torch.cuda.is_available():
        checkpoint = torch.load(PATH)
    else:
        # Load GPU model on CPU
        checkpoint = torch.load(PATH,
                                map_location=lambda storage,
                                                    loc: storage)
    return checkpoint


# def visualize_model(model, dataloaders, device, class_names, num_images=10):
#     was_training = model.training
#     model.eval()
#     images_so_far = 0
#     fig = plt.figure()
#
#     with torch.no_grad():
#         for i, (inputs, labels) in enumerate(dataloaders['val']):
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#
#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)
#             probs = [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, outputs)]
#             for j in range(inputs.size()[0]):
#                 if preds[j] == labels[j].item():
#                     continue
#                 images_so_far += 1
#                 ax = plt.subplot(num_images // 2, 2, images_so_far)
#                 ax.axis('off')
#                 ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
#                     class_names[preds[j]],
#                     probs[j] * 100.0,
#                     class_names[labels[j]]), color=("green" if preds[j] == labels[j].item() else "red"))
#                 # ax.set_title('predicted: {}'.format(class_names[preds[j]]))
#                 matplotlib_imshow(inputs.cpu().data[j], one_channel=True)
#                 # plt.imread(inputs.cpu().data[j])
#                 if images_so_far == num_images:
#                     model.train(mode=was_training)
#                     return fig
#         model.train(mode=was_training)
#         return fig

def visualize_model(model, dataloaders, device, class_names, num_images=10, show_only_bad=True):
    was_training = model.training
    model.eval()
    images_so_far = 0
    img_pr_row = 5
    fig = plt.figure(figsize=(20, 20))
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            probs = [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, outputs)]
            row = math.ceil(num_images / img_pr_row)
            for j in range(inputs.size()[0]):
                if show_only_bad and preds[j] == labels[j].item():
                    continue
                images_so_far += 1
                ax = fig.add_subplot(row, img_pr_row, images_so_far, xticks=[], yticks=[])
                matplotlib_imshow(inputs[j], one_channel=True)
                ax.axis('off')
                ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
                    class_names[preds[j]],
                    probs[j] * 100.0,
                    class_names[labels[j]]), color=("green" if preds[j] == labels[j].item() else "red"))

                # ax = plt.subplot(num_images // 2, 2, images_so_far)
                #
                # # ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                # matplotlib_imshow(inputs[j], one_channel=True)
                # # plt.imread(inputs.cpu().data[j])
                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return fig
        model.train(mode=was_training)
        return fig


def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    with torch.no_grad():
        output = net(images)
        # convert output probabilities to predicted class
        _, preds_tensor = torch.max(output, 1)
        preds = np.squeeze(preds_tensor.cpu().numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.cpu().numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def plot_classes_preds(model, images, labels, classes, batch_size):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(model, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(20, 20))
    for idx in np.arange(min(min(batch_size, 8), len(images))):
        ax = fig.add_subplot(2, 4, idx + 1, xticks=[], yticks=[])
    matplotlib_imshow(images[idx], one_channel=True)
    ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
        classes[preds[idx]],
        probs[idx] * 100.0,
        classes[labels[idx]]),
        color=("green" if preds[idx] == labels[idx].item() else "red"))
    return fig


def run_training(data_dir, output_dir, model_name, num_classes, num_epochs, batch_size, feature_extract,
                 use_pretrained):
    # Initialize the model for this run
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=use_pretrained)

    # Print the model we just instantiated
    # print(model_ft)

    ######################################################################
    # Load Data
    # ---------
    #
    # Now that we know what the input size must be, we can initialize the data
    # transforms, image datasets, and the dataloaders. Notice, the models were
    # pretrained with the hard-coded normalization values, as described
    # `here <https://pytorch.org/docs/master/torchvision/models.html>`__.
    #

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = get_data_transform(model_name, input_size)

    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    # Create training and validation dataloaders
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in
        ['train', 'val']}
    class_names = image_datasets['train'].classes
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    ######################################################################
    # if output dir contains checkpoint, load checkpoint and insert into model
    output_dir_path = Path(output_dir)
    if not output_dir_path.exists():
        output_dir_path.mkdir(parents=True)
    output_file = Path(output_dir_path, "checkpoint.pth.tar")
    current_epoch = 0
    loss = 1
    optimizer_state = None
    current_best_acc = 0
    if not output_file.exists():
        debug = 0
    else:
        checkpoint = load_model_data(str(output_file))
        current_epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]
        model_ft.load_state_dict(checkpoint['model_state_dict'])
        optimizer_state = checkpoint["optimizer_state_dict"]
        # current_best_acc = 0.82
        current_best_acc = checkpoint["accuracy"]
        print("Loaded: (Loss:", str(loss), ") (best acc:", str(current_best_acc), ") (current epoch:",
              str(current_epoch), ")")

    ######################################################################
    # Create the Optimizer
    # --------------------
    #
    # Now that the model structure is correct, the final step for finetuning
    # and feature extracting is to create an optimizer that only updates the
    # desired parameters. Recall that after loading the pretrained model, but
    # before reshaping, if ``feature_extract=True`` we manually set all of the
    # parameter’s ``.requires_grad`` attributes to False. Then the
    # reinitialized layer’s parameters have ``.requires_grad=True`` by
    # default. So now we know that *all parameters that have
    # .requires_grad=True should be optimized.* Next, we make a list of such
    # parameters and input this list to the SGD algorithm constructor.
    #
    # To verify this, check out the printed parameters to learn. When
    # finetuning, this list should be long and include all of the model
    # parameters. However, when feature extracting this list should be short
    # and only include the weights and biases of the reshaped layers.
    #

    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)
    if len(params_to_update) == 0:
        params_to_update = model_ft.parameters()
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=0.1, momentum=0.9)
    if optimizer_state is not None:
        optimizer_ft.load_state_dict(optimizer_state)
    ######################################################################
    # Run Training and Validation Step
    # --------------------------------
    #
    # Finally, the last step is to setup the loss for the model, then run the
    # training and validation function for the set number of epochs. Notice,
    # depending on the number of epochs this step may take a while on a CPU.
    # Also, the default learning rate is not optimal for all of the models, so
    # to achieve maximum accuracy it would be necessary to tune for each model
    # separately.
    #

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()
    if current_epoch == 0:
        last_epoch = -1
    else:
        current_epoch += 1
        last_epoch = current_epoch
    scheduler = lr_scheduler.ExponentialLR(optimizer_ft, gamma=0.98, last_epoch=last_epoch)
    # Train and evaluate

    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, scheduler, class_names,
                                 loss=loss,
                                 device=device,
                                 output_file=output_file,
                                 num_epochs=num_epochs, current_epoch=current_epoch,
                                 is_inception=(model_name == "inception"), best_accuracy=current_best_acc,
                                 batch_size=batch_size)

    # TODO: SAVE THIS MODEL:


if __name__ == '__main__':
    print("PyTorch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help="""\
   This is the path to the folder containing the image data.\
       Top level data directory. Here we assume the format of the directory conforms \
       to the ImageFolder structure
   """
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help="""\
       This is the path to the output folder, where the model will be saved.\
       """
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default="inception",
        help="""\
   The name of the model to train.\
   Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]\
   """
    )
    parser.add_argument(
        '--use_pretrained_model',
        type=bool,
        default=True,
        help="""\
           this is the procent moved files, have to be between 0.001 and 1.\
           """
    )
    parser.add_argument(
        '--preprocess_data',
        type=bool,
        default=False,
        help="""\
       Resize, crop, grayscale, all the good stuff.\
       """
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        required=True,
        help="""\
   Number of classes in the dataset\
   """
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help="""\
       Batch size for training (change depending on how much memory you have)\
       """
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=0,
        help="""\
       Number of epochs to train for,\
       if none given it will go forever or until manually canceled\
       """
    )
    parser.add_argument(
        '--use_feature_extract',
        type=bool,
        default=False,
        help="""\
           Flag for feature extracting. When False, we finetune the whole model,\
            when True we only update the reshaped layer params.\
           """
    )

    FLAGS, unparsed = parser.parse_known_args()

    # Top level data directory. Here we assume the format of the directory conforms
    #   to the ImageFolder structure
    data_dir = FLAGS.data_dir

    # The output dir, the training will save the model while training so it can be resumed later on.
    output_dir = FLAGS.output_dir

    # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
    model_name = FLAGS.model_name

    # Number of classes in the dataset
    assert FLAGS.num_classes > 0
    num_classes = FLAGS.num_classes

    # Batch size for training (change depending on how much memory you have)
    assert FLAGS.batch_size > 0
    batch_size = FLAGS.batch_size

    # Number of epochs to train for
    assert FLAGS.num_epochs >= 0
    if FLAGS.num_epochs == 0:
        num_epochs = sys.maxsize
        print("number of epochs is set to,", str(num_epochs))
    else:
        num_epochs = FLAGS.num_epochs

    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    feature_extract = FLAGS.use_feature_extract

    # If the model we load should have pretrained weight. usually used for fine tuning.
    use_pretrained = FLAGS.use_pretrained_model

    # Is set to true if we should preprocess the data before training.
    preprocess_data = FLAGS.preprocess_data
    run_training(data_dir, output_dir, model_name, num_classes, num_epochs, batch_size, feature_extract,
                 use_pretrained)
