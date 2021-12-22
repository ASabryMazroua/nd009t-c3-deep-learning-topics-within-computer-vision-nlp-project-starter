#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import argparse
import logging
import sys
import os

#We'll start the logger where we'll add logging info in each step
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def test(model, test_loader, criterion, device):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    logger.info(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)))

def train(model, train_loader, validation_loader, criterion, optimizer, device):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    epochs = 30 #Setting the number of epochs
    
    for epoch in range(1, epochs+1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader, 1):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                logger.info(
                    "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
        test(model, validation_loader, criterion)
    return model
    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet101(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False #Freezing the ResNet layers not to train them
    output_shape = model.fc.in_features #Getting the output shape of the ResNet
    model.fc = nn.Sequential(
                   nn.Linear(output_shape, 1024),
                   nn.ReLU(inplace = True),
                   nn.Linear(1024, 512),
                   nn.ReLU(inplace = True),
                   nn.Linear(512, 133))
    return model

def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    #Preparing the paths for each flder 
    train_path = os.path.join(data, 'train')
    test_path = os.path.join(data, 'test')
    validation_path=os.path.join(data, 'valid')
    
    #All pre-trained models expect input images normalized in the same way
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    # Integrating all transformations and augmentation steps for train and test data
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), 
                                       transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       normalize])
                                                            
    test_transform = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      normalize])

    #Creating the different data loaders
    train_data = torchvision.datasets.ImageFolder(root=train_path, transform=train_transform)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    validation_data = torchvision.datasets.ImageFolder(root=validation_path, transform=test_transform)
    validation_data_loader  = torch.utils.data.DataLoader(validation_data, batch_size=batch_size) 
    
    test_data = torchvision.datasets.ImageFolder(root=test_path, transform=test_transform)
    test_data_loader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
    
    return train_data_loader, validation_data_loader, test_data_loader

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    #Enabling the GPU to minimize the run time and cost
    #Source: https://sagemaker-examples.readthedocs.io/en/latest/prep_data/image_data_guide/04c_pytorch_training.html
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model=net()
    model=model.to(device)
    
    '''
    TODO: Create your loss and optimizer
    '''    
    loss_criterion = nn.CrossEntropyLoss() #We'll use the Cross Entropy as a Loss Function
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr) #Using Adam as our optimizer
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    train_loader, validation_loader, test_loader = create_data_loaders(args.data_dir, args.batchsize) #Creating data loaders
    logger.info("Training the model") #Adding infor to mark the training start
    model = train(model, train_loader, validation_loader, loss_criterion, optimizer, device) #Training the model
    
    '''
    TODO: Test the model to see its accuracy
    '''
    logger.info("Testing the model")
    test(model, test_loader, loss_criterion, device)
    
    '''
    TODO: Save the trained model
    '''
    logger.info("Saving the model")
    model = model.state_dict()
    path = os.path.join(args.model_dir, "model.pth")
    torch.save(model, path)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    
    parser.add_argument(
        "--batchsize",
        type = int,
        default = 64,
        metavar = "N",
        help = "input batch size for training (default: 64)",
    )
    
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )
    
    parser.add_argument('--data_dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    
    parser.add_argument('--model_-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    
    args = parser.parse_args()

    main(args)
