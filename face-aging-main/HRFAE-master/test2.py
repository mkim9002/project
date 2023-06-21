import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import yaml

from PIL import Image
from torchvision import transforms, utils

from datasets import *
from nets import *
from functions import *
from trainer import *
import time

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None

start = time.time()

if torch.cuda.is_available():
    device = torch.device('cuda')
    num_devices = torch.cuda.device_count()
else:
    device = torch.device('cpu')
    num_devices = 1

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='001', help='path to the config file.')
parser.add_argument('--vgg_model_path', type=str, default='./models/dex_imdb_wiki.caffemodel.pt', help='pretrained age classifier')
parser.add_argument('--log_path', type=str, default='./logs/', help='log file path')
parser.add_argument('--checkpoint', type=str, default='', help='checkpoint file path')
parser.add_argument('--img_path', type=str, default='./test/input/', help='test image path')
parser.add_argument('--out_path', type=str, default='./test/output/', help='test output path')
parser.add_argument('--target_age', type=int, default=65, help='Age transform target, interger value between 20 and 70')
parser.add_argument('--min_age', type=int, default=20, help='Age transform minValue, interger value between 20 and 70')
parser.add_argument('--max_age', type=int, default=70, help='Age transform maxValue, interger value between 20 and 70')
parser.add_argument('--age_sep', type=int, default=1, help='Age transform interval, interger value between 1 and 10')
opts = parser.parse_args()

log_dir = os.path.join(opts.log_path, opts.config) + '/'
if not os.path.exists(opts.out_path):
    os.makedirs(opts.out_path)

config = yaml.safe_load(open('c:/project/face-aging-main/configs/' + opts.config + '.yaml', 'r'))

img_size = (config['input_w'], config['input_h'])

# Initialize trainer
trainer = Trainer(config)

# Load pretrained model
if opts.checkpoint:
    trainer.load_checkpoint(opts.checkpoint)
else:
    trainer.load_checkpoint(log_dir + 'checkpoint')

# Move the trainer to the device
trainer = trainer.to(device)

# Parallelize the trainer across multiple GPUs
if num_devices > 1:
    trainer = nn.DataParallel(trainer)

# Set target age
target_age = opts.target_age

# Load test image
img_list = os.listdir(opts.img_path)
img_list.sort()

# Preprocess function
def preprocess(img_name):
    resize = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor()
    ])
    normalize = transforms.Normalize(mean=[0.48501961, 0.45795686, 0.40760392], std=[1, 1, 1])
    img_pil = Image.open(opts.img_path + img_name)
    img_np = np.array(img_pil)
    img = resize(img_pil)
    if img.size(0) == 1:
        img = torch.cat((img, img, img), dim=0)
    img = normalize(img)
    return img

# Testing loop
with torch.no_grad():
    for img_name in img_list:
        if not img_name.endswith(('png', 'jpg', 'PNG', 'JPG')):
            print('File ignored: ' + img_name)
            continue
        image_A = preprocess(img_name)
        image_A = image_A.unsqueeze(0).to(device)

        min_age = opts.min_age
        max_age = opts.max_age
        age_sep = opts.age_sep
        target_age_list = list(range(min_age, max_age+1, age_sep))
        if (max_age - min_age) % age_sep != 0:
            target_age_list.append(max_age)
        
        # Create a list to store the modified images
        modified_images = []
        
        for tg in target_age_list:
            age_modif = torch.tensor(tg).unsqueeze(0).to(device)
            
            # Perform age modification using the trainer
            image_A_modif = trainer.module.test_eval(image_A, age_modif, target_age=tg, hist_trans=True) if num_devices > 1 else trainer.test_eval(image_A, age_modif, target_age=tg, hist_trans=True)
            
            modified_images.append(image_A_modif)
        
        # Save the modified images
        for i, tg in enumerate(target_age_list):
            utils.save_image(clip_img(modified_images[i]), opts.out_path + img_name.split('.')[0] + '_age_' + str(tg) + '.jpg')
end = time.time()
print(f'걸린시간 : {round(end-start,2)}')