from pathlib import Path

import matplotlib.pyplot as plt
import PIL
from PIL import Image
from torchvision.transforms import transforms

if __name__ == '__main__':
    ip = Path("/media/linux/VOID/code/data/DoubleBag/eval_results/inception_jfs")
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    for img in ip.glob("*.jpg"):
        image = Image.open(str(img))
        image = image.convert('RGB')
        image_tensor = data_transform(image).float()
        image = transforms.ToPILImage()(image_tensor)
        # image = transforms.ToPILImage(image_tensor).convert("RGB")
        image.show()
        break