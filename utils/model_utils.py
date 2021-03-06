from __future__ import division
from __future__ import print_function

import torch.nn as nn
from cnn_finetune import make_model
from torchvision import datasets, models, transforms


######################################################################
# Set Model Parameters’ .requires_grad attribute
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# This helper function sets the ``.requires_grad`` attribute of the
# parameters in the model to False when we are feature extracting. By
# default, when we load a pretrained model all of the parameters have
# ``.requires_grad=True``, which is fine if we are training from scratch
# or finetuning. However, if we are feature extracting and only want to
# compute gradients for the newly initialized layer then we want all of
# the other parameters to not require gradients. This will make more sense
# later.
#

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def get_data_transform(model_name, input_size):
    if model_name == "inceptionresnetv2" or model_name == 'pnasnet5large':
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(input_size),
                # transforms.RandomResizedCrop(input_size, scale=(0.8, 1)),
                # transforms.RandomErasing(p=0.2),
                # transforms.Grayscale(3),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
            'val': transforms.Compose([
                transforms.Resize(input_size),
                # transforms.CenterCrop(input_size),
                # transforms.Grayscale(3),
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
        }
    else:
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(input_size),
                # transforms.RandomResizedCrop(input_size),
                # transforms.RandomVerticalFlip(),
                # transforms.RandomErasing,
                transforms.Grayscale(3),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(input_size),
                # transforms.CenterCrop(input_size),
                transforms.Grayscale(3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
    return data_transforms


######################################################################
# Initialize and Reshape the Networks
# -----------------------------------
#
# Now to the most interesting part. Here is where we handle the reshaping
# of each network. Note, this is not an automatic procedure and is unique
# to each model. Recall, the final layer of a CNN model, which is often
# times an FC layer, has the same number of nodes as the number of output
# classes in the dataset. Since all of the models have been pretrained on
# Imagenet, they all have output layers of size 1000, one node for each
# class. The goal here is to reshape the last layer to have the same
# number of inputs as before, AND to have the same number of outputs as
# the number of classes in the dataset. In the following sections we will
# discuss how to alter the architecture of each model individually. But
# first, there is one important detail regarding the difference between
# finetuning and feature-extraction.
#
# When feature extracting, we only want to update the parameters of the
# last layer, or in other words, we only want to update the parameters for
# the layer(s) we are reshaping. Therefore, we do not need to compute the
# gradients of the parameters that we are not changing, so for efficiency
# we set the .requires_grad attribute to False. This is important because
# by default, this attribute is set to True. Then, when we initialize the
# new layer and by default the new parameters have ``.requires_grad=True``
# so only the new layer’s parameters will be updated. When we are
# finetuning we can leave all of the .required_grad’s set to the default
# of True.
#
# Finally, notice that inception_v3 requires the input size to be
# (299,299), whereas all of the other models expect (224,224).
#
# Resnet
# ~~~~~~
#
# Resnet was introduced in the paper `Deep Residual Learning for Image
# Recognition <https://arxiv.org/abs/1512.03385>`__. There are several
# variants of different sizes, including Resnet18, Resnet34, Resnet50,
# Resnet101, and Resnet152, all of which are available from torchvision
# models. Here we use Resnet18, as our dataset is small and only has two
# classes. When we print the model, we see that the last layer is a fully
# connected layer as shown below:
#
# ::
#
#    (fc): Linear(in_features=512, out_features=1000, bias=True)
#
# Thus, we must reinitialize ``model.fc`` to be a Linear layer with 512
# input features and 2 output features with:
#
# ::
#
#    model.fc = nn.Linear(512, num_classes)
#
# Alexnet
# ~~~~~~~
#
# Alexnet was introduced in the paper `ImageNet Classification with Deep
# Convolutional Neural
# Networks <https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf>`__
# and was the first very successful CNN on the ImageNet dataset. When we
# print the model architecture, we see the model output comes from the 6th
# layer of the classifier
#
# ::
#
#    (classifier): Sequential(
#        ...
#        (6): Linear(in_features=4096, out_features=1000, bias=True)
#     )
#
# To use the model with our dataset we reinitialize this layer as
#
# ::
#
#    model.classifier[6] = nn.Linear(4096,num_classes)
#
# VGG
# ~~~
#
# VGG was introduced in the paper `Very Deep Convolutional Networks for
# Large-Scale Image Recognition <https://arxiv.org/pdf/1409.1556.pdf>`__.
# Torchvision offers eight versions of VGG with various lengths and some
# that have batch normalizations layers. Here we use VGG-11 with batch
# normalization. The output layer is similar to Alexnet, i.e.
#
# ::
#
#    (classifier): Sequential(
#        ...
#        (6): Linear(in_features=4096, out_features=1000, bias=True)
#     )
#
# Therefore, we use the same technique to modify the output layer
#
# ::
#
#    model.classifier[6] = nn.Linear(4096,num_classes)
#
# Squeezenet
# ~~~~~~~~~~
#
# The Squeeznet architecture is described in the paper `SqueezeNet:
# AlexNet-level accuracy with 50x fewer parameters and <0.5MB model
# size <https://arxiv.org/abs/1602.07360>`__ and uses a different output
# structure than any of the other models shown here. Torchvision has two
# versions of Squeezenet, we use version 1.0. The output comes from a 1x1
# convolutional layer which is the 1st layer of the classifier:
#
# ::
#
#    (classifier): Sequential(
#        (0): Dropout(p=0.5)
#        (1): Conv2d(512, 1000, kernel_size=(1, 1), stride=(1, 1))
#        (2): ReLU(inplace)
#        (3): AvgPool2d(kernel_size=13, stride=1, padding=0)
#     )
#
# To modify the network, we reinitialize the Conv2d layer to have an
# output feature map of depth 2 as
#
# ::
#
#    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
#
# Densenet
# ~~~~~~~~
#
# Densenet was introduced in the paper `Densely Connected Convolutional
# Networks <https://arxiv.org/abs/1608.06993>`__. Torchvision has four
# variants of Densenet but here we only use Densenet-121. The output layer
# is a linear layer with 1024 input features:
#
# ::
#
#    (classifier): Linear(in_features=1024, out_features=1000, bias=True)
#
# To reshape the network, we reinitialize the classifier’s linear layer as
#
# ::
#
#    model.classifier = nn.Linear(1024, num_classes)
#
# Inception v3
# ~~~~~~~~~~~~
#
# Finally, Inception v3 was first described in `Rethinking the Inception
# Architecture for Computer
# Vision <https://arxiv.org/pdf/1512.00567v1.pdf>`__. This network is
# unique because it has two output layers when training. The second output
# is known as an auxiliary output and is contained in the AuxLogits part
# of the network. The primary output is a linear layer at the end of the
# network. Note, when testing we only consider the primary output. The
# auxiliary output and primary output of the loaded model are printed as:
#
# ::
#
#    (AuxLogits): InceptionAux(
#        ...
#        (fc): Linear(in_features=768, out_features=1000, bias=True)
#     )
#     ...
#    (fc): Linear(in_features=2048, out_features=1000, bias=True)
#
# To finetune this model we must reshape both layers. This is accomplished
# with the following
#
# ::
#
#    model.AuxLogits.fc = nn.Linear(768, num_classes)
#    model.fc = nn.Linear(2048, num_classes)
#
# Notice, many of the models have similar output structures, but each must
# be handled slightly differently. Also, check out the printed model
# architecture of the reshaped network and make sure the number of output
# features is the same as the number of classes in the dataset.

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3 
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299
    elif model_name == "inceptionresnetv2":
        model_ft = make_model(model_name, num_classes=num_classes, dropout_p=0.3)
        # set_parameter_requires_grad(model_ft, feature_extract)
        input_size = 224
    elif model_name == 'pnasnet5large':
        model_ft = make_model(model_name, num_classes=num_classes, dropout_p=0.5)
        print(model_ft.original_model_info)
        input_size = 331
    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size
