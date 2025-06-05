import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import random

class PairedRandomCrop:
    def __init__(self, size):
        self.size = size
        
    def __call__(self, img, background, mask):
        assert img.size == mask.size == background.size, "All inputs must have the same size"

        w, h = img.size
        th, tw = self.size, self.size
        
        # if input size < target size
        if w < tw or h < th:
            scale = max(tw / w, th / h) * 1.1
            new_size = (int(w * scale), int(h * scale))
            img = F.resize(img, new_size, interpolation=transforms.InterpolationMode.BILINEAR)
            background = F.resize(background, new_size, interpolation=transforms.InterpolationMode.BILINEAR)
            mask = F.resize(mask, new_size, interpolation=transforms.InterpolationMode.NEAREST)
            w, h = new_size

        if w == tw and h == th:
            return img, background, mask
            
        # calculate random crop position
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        
        # allpy crop to all images
        return (F.crop(img, i, j, th, tw), 
                F.crop(background, i, j, th, tw), 
                F.crop(mask, i, j, th, tw))