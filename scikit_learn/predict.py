import torch
import warnings
from PIL import Image
from torchvision import transforms
from torchvision import datasets, models, transforms
#from torchsummary import summary
from paradox.extension.Scikit_Learn.helpers import *

def image_transform(imagepath):
    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    image = Image.open(imagepath)
    imagetensor = test_transforms(image)
    imagetensor = imagetensor.unsqueeze(0)
    return imagetensor
 
# Preprocess the image before feeding into model
def load_datasets(datasets_path):
    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(datasets_path, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                 shuffle=True, num_workers=4)
                  for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
    return dataloaders,dataset_sizes,class_names,device

# function for predication 
def predict(image_path, model_path,label):
    # Predication using saved model
    # PATH = "./models/dogcat_model.pt"

    # dataset_path = "./datasets/cats_dogs/"

    model_path = model_path
    # dataset_path = dataset_path
    image_path = image_path

    # Load
    model = torch.load(model_path)
    model.eval()


    # img_path = './test_images/cat.4037.jpg'
    input = image_transform(image_path)


    outputs = model(input)
    _, preds = torch.max(outputs, 1)

    # converting the predication to scalar
    preds = preds.numpy()
    preds = preds.item()


    # attach label to the predication class

    # dataloaders,dataset_sizes,class_name,device = load_datasets(dataset_path)
    label = label
    
    result = ""
    # classifiying the predication class
    if (preds!=0):
        print(label[1])
        result = label[1]
    else:
        print(label[0])
        result = label[0] 
    return result
