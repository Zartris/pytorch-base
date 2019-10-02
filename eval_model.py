import argparse
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import torch
from PIL import Image
from torchvision import datasets, transforms
from utils.model_utils import initialize_model


def get_dataloader(data_dir, model_name, input_size):
    data_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    image_Folder = datasets.ImageFolder(data_dir, data_transform)
    data_loader = torch.utils.data.DataLoader(image_Folder, batch_size=64, shuffle=False, num_workers=4)
    # Create training and validation dataloaders

    class_names = image_Folder.classes
    return data_transform, class_names, data_loader


def predict_image(model, image_path, output_path, label, data_transform, data_loader):
    if label == "2bags":
        debug = 0
    image = Image.open(str(image_path))
    # image = image.convert('RGB')
    image_tensor = data_transform(image).float()
    # matplotlib_imshow(image_tensor, True)
    # plt.show()
    image_batch = image_tensor.unsqueeze_(0)
    image_batch = image_batch.to(device)
    with torch.no_grad():
        output = model.forward(image_batch)
    output = torch.exp(output)
    top_p, top_class = output.topk(1, dim=1)
    # print(top_p, top_class)
    # _, index = torch.max(output.data, 1)
    # index = output.data.cpu().numpy().argmax()
    real_label = ""
    if class_name[top_class.item()] == '1bag':
        if image_path.name not in label:
            # print("TRUE: guessed:", str(class_name[top_class.item()]), "label:", str(label))
            real_label = "1bag"
        else:
            shutil.copyfile(str(image_path), str(Path(output_path, "1bag", image_path.name)))
            real_label = "2bags"
            # print("FALSE: guessed:", str(class_name[top_class.item()]), "label:", str(label), str(image_path))
    elif class_name[top_class.item()] == '2bags':
        if image_path.name in label:
            # print("TRUE: guessed:", str(class_name[top_class.item()]), "label:", str(label))
            real_label = "2bags"
        else:
            shutil.copyfile(str(image_path), str(Path(output_path, "2bags", image_path.name)))
            real_label = "1bag"
            # print("FALSE: guessed:", str(class_name[top_class.item()]), "label:", str(label), str(image_path))
    return class_name[top_class.item()], real_label


def load_model_data(PATH):
    print("=> loading checkpoint '{}' ...".format(PATH))
    if torch.cuda.is_available():
        checkpoint = torch.load(PATH)
    else:
        # Load GPU model on CPU
        checkpoint = torch.load(PATH,
                                map_location=lambda storage,
                                                    loc: storage)
    return checkpoint["model_state_dict"]


def eval_data(model, data_loader, class_name):
    model.eval()
    with torch.no_grad():
        for inputs, labels in data_loader:
            # Move to device
            inputs, labels = inputs.to(device), labels.to(device)
            # Forward pass
            output = model.forward(inputs)

            # Since our model outputs a LogSoftmax, find the real
            # percentages by reversing the log function
            output = torch.exp(output)
            # Get the top class of the output
            top_p, top_class = output.topk(1, dim=1)
            # See how many of the classes were correct?
            equals = top_class == labels.view(*top_class.shape)
            for index, input in enumerate(inputs):
                if not equals[index].item():
                    matplotlib_imshow(input, True)
                    plt.show()
            print(equals)


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.cpu().numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


if __name__ == '__main__':
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
        This is the path to the folder containing the image data.\
            Top level data directory. Here we assume the format of the directory conforms \
            to the ImageFolder structure
        """
    )
    parser.add_argument(
        '--model_file',
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
        '--num_classes',
        type=int,
        required=True,
        help="""\
    Number of classes in the dataset\
    """
    )

    FLAGS, unparsed = parser.parse_known_args()
    data_dir = Path(FLAGS.data_dir)
    output_dir = Path(FLAGS.output_dir)
    model_file = FLAGS.model_file
    model_name = FLAGS.model_name
    num_classes = FLAGS.num_classes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_ft, input_size = initialize_model(model_name, num_classes, True, True)
    model_ft.load_state_dict(load_model_data(model_file))
    model_ft.eval()
    model_ft.to(device)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    # data_transform, class_name, data_loader = get_dataloader(data_dir, model_name, input_size)
    # # eval_data(model_ft, data_loader, class_name)
    # for folder in data_dir.glob("*"):
    #     for img_path in folder.glob("*.jpg"):
    #         predict_image(model_ft, img_path, output_dir, str(folder.stem), data_transform, class_name)

    # Hacking:
    onebag_dir = Path(output_dir, "1bag")
    if onebag_dir.exists():
        onebag_dir.rmdir()
    onebag_dir.mkdir(parents=True)

    twobags_dir = Path(output_dir, "2bags")
    if twobags_dir.exists():
        twobags_dir.rmdir()
    twobags_dir.mkdir(parents=True)

    twobags = []
    class_name = ['1bag', '2bags']
    data_transform = transforms.Compose([
        transforms.Resize(input_size),
        # transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    for image in Path('/media/linux/VOID/code/data/DoubleBag/eval_results/Manual').glob("*.jpg"):
        twobags.append(image.name)
    rounds = len(list(data_dir.glob('*.jpg')))
    counter = 0
    for image in data_dir.glob('*.jpg'):
        counter += 1
        guessed_label, true_label = predict_image(model_ft, image, output_dir, twobags, data_transform, class_name)
        print(str(counter) + "/" + str(rounds) + ":", str(guessed_label == true_label), " - guessed:", str(guessed_label),
              "was:", true_label)
