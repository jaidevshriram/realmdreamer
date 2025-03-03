import torchvision.transforms as transforms
from PIL import Image


def load_image_to_pt(img_path, h=512, w=512):

    img = Image.open(img_path).resize((h, w)).convert("RGB")
    img_pt = transforms.ToTensor()(img).unsqueeze(0)

    return img_pt  # B C H W
