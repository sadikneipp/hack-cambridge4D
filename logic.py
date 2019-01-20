import torch
from torch import nn
import torchvision.models as models
import os 
from PIL import Image
from torchvision import models, transforms
import base64
from importlib import import_module
from acquisition import camera
from io import BytesIO

PATH = 'model/dumps' 
file = os.listdir(PATH)[0]
filepath = PATH + '/' + file

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)

preprocess = transforms.Compose([
   transforms.Scale(224),
   transforms.ToTensor(),
   normalize
])

upsample = nn.Upsample((224, 224))

imagepath = '/home/sadikneipp/Desktop/cambridge_hack/acquisition/dataset/ibuprofen/1547929905_0.jpg'

classes = sorted(os.listdir('acquisition/dataset'))

def model_load():
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 4)
    
    saved_state = torch.load(filepath, map_location=lambda storage, loc: storage)
    model_ft.load_state_dict(saved_state)
    model_ft.eval()
    return model_ft

def preprocess_pil(img_pil):
    img_tensor = preprocess(img_pil)
    img_tensor.unsqueeze_(0)
    img_tensor = upsample(img_tensor)
    return img_tensor
    
def predict(model_ft):
    img_pil = Image.open('image.jpg')
    img_tensor = preprocess_pil(img_pil)
    fc_out = model_ft(img_tensor)
    _, indices = torch.max(fc_out, 1)
    return classes[indices[0]] 

if __name__ == '__main__':
    im = Image.open(imagepath)
    buffered = BytesIO()
    im.save(buffered, format="JPEG")
    b64_im = base64.b64encode(buffered.getvalue())
    model = model_load()
    print (predict(model, b64_im))
