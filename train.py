import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
import argparse
import numpy as np
import random
from tqdm import tqdm
from loguru import logger
from models_res import ResNet, SkipBlock
from data_loader import save_plots, get_data
parser = argparse.ArgumentParser()


class ResNet18_baseline(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base = resnet18(weights=None, num_classes=num_classes)
        self.base = nn.Sequential(*list(base.children())[:-1])
        in_features = base.fc.in_features
        self.drop = nn.Dropout()
        self.final = nn.Linear(in_features,num_classes)
    
    def forward(self,x):
        x = self.base(x)
        x = self.drop(x.view(-1,self.final.in_features))
        return self.final(x)


# Training function.
def train(model, trainloader, optimizer, criterion, device):
    model.train()
    logger.info('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        outputs = model(image)
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        loss.backward()
        optimizer.step()

    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc

def validate(model, testloader, criterion, device):
    model.eval()
    logger.info('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            outputs = model(image)
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()
        
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc



parser.add_argument(
    '-m', '--model', default='scratch',
    help='choose model built from scratch or the Torchvision model',
    choices=['scratch', 'torchvision', 'relearn']
)
parser.add_argument(
    "-o", "--original_model_path",
    help="path of original model to get target acc for relarn time"
)
parser.add_argument(
    "-u", "--unlearned_model_path",
    help="path of unlearned model to calculate relearn time from"
)
args = parser.parse_args()

if args.original_model_path is not None or args.unlearned_model_path is not None:
    assert args.model == "relearn", "Model paths are for calculating relearn time; see help"
# Set seed.
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

epochs = 20
batch_size = 64
learning_rate = 0.01
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_loader, valid_loader = get_data(batch_size=batch_size)
criterion = nn.CrossEntropyLoss()
# Define model based on the argument parser string.
if args.model == 'scratch':
    logger.info('Training ResNet18 built from scratch...')
    model = ResNet(n_channels=3, n_layers=18, block=SkipBlock, n_classes=10).to(device)
    plot_name = 'resnet_scratch'
if args.model == 'torchvision':
    logger.info('Training the Torchvision ResNet18 model...')
    model = build_model(pretrained=False, fine_tune=True, num_classes=10).to(device) 
    plot_name = 'resnet_torchvision'
if args.model == 'relearn':
    logger.info("Calculating relearn time")
    original_model = ResNet18_baseline(num_classes=10).to(device)
    original_model.load_state_dict(torch.load(args.original_model_path, map_location=device))
    
    _, original_valid_acc = validate(
        original_model, 
        valid_loader, 
        criterion,
        device
    )
    logger.info(f"Original val acc: {original_valid_acc}")
    
    model = ResNet18_baseline(num_classes=10).to(device)
    model.load_state_dict(torch.load(args.unlearned_model_path, map_location=device))
    

# Total parameters and trainable parameters.
total_params = sum(p.numel() for p in model.parameters())
logger.info(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
logger.info(f"{total_trainable_params:,} training parameters.")


optimizer = optim.SGD(model.parameters(), lr=learning_rate)


if __name__ == '__main__':
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_acc = train(
            model, 
            train_loader, 
            optimizer, 
            criterion,
            device
        )
        valid_epoch_loss, valid_epoch_acc = validate(
            model, 
            valid_loader, 
            criterion,
            device
        )
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        logger.info(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        logger.info(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
        logger.info('-'*50)
        if args.model == 'relearn' and valid_epoch_acc >= original_valid_acc:
            logger.info(f"Relearn time for {args.unlearned_model_path} is {epoch}")
            break
        
    # Save the loss and accuracy plots.
    save_plots(
        train_acc, 
        valid_acc, 
        train_loss, 
        valid_loss, 
        name=plot_name
    )
    logger.info('TRAINING COMPLETE')