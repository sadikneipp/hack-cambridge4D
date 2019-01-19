import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms
import torchvision.models as models
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import random_split
import copy
from torch import nn
import time
import argparse
import pickle
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

#Environment Parameters
COLAB = False

BS = 32
WORKERS= 2
N_SPLITS = 10 # number of times to split dataset randomly and fit
SPLIT_MODE = 'auto' #auto will use the torch dataset random splitter, manual will use the premade folder splits
VAL_SIZE = 0.2 #percentage of validation data
OUTPUT_PATH = '../../gdrive/My Drive/BI_UCL/train_data/'

''' 
Training parameters 

Follows pytorch naming conventions   
'''

LR_0 = 0.001 #initial learning rate
GAMMA = 0.1
N_EPOCHS = 15 #total epochs
STEP_SIZE = 3 #epochs until decay of lr by gamma
MOMENTUM = 0.9
WD = 5e-3 #weight decay

def get_args(): #deprecated for colab e.e
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", help="pass 'gpu' to turn on GPU")
    args = parser.parse_args()

    return args

def check_create(path):
    if os.path.exists(path):
        print(path + ' exists')
        return
        
    os.makedirs(path)
    print(path + ' created')

def save_data(data):
    check_create(OUTPUT_PATH)
    now = str(int(time.time()))
    with open(OUTPUT_PATH + now + '.pickle', 'wb') as handle:
        pickle.dump(data, handle)

def balanced_valid_ixs(valid):
    crackle_ix = []
    no_crackle_ix = []
    
    for i in range(len(valid)):
        if valid[i]['label'].item() == 1:
            crackle_ix.append(i)
        else:
            no_crackle_ix.append(i)
    
    return crackle_ix + no_crackle_ix[:len(crackle_ix)]

def init_loaders():
    tfs = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    if SPLIT_MODE == 'auto':
        whole = datasets.ImageFolder('../acquisition/dataset', transform=tfs)
        n_train = int(len(whole)*(1-VAL_SIZE))
        n_val = len(whole) - n_train
        train, valid = random_split(whole, [n_train, n_val])

    trainloader = DataLoader(train, batch_size=BS, drop_last=False)

    validloader = DataLoader(valid, batch_size=BS, drop_last=False,
                            shuffle=False)

    return trainloader, validloader

def metrics(targets, y_pred):
    return torch.sum(targets == y_pred).item() / targets.numel(), confusion_matrix(targets.numpy()
                                                        , y_pred.numpy())
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    # upsample = nn.Upsample((224, 224))
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    # data = {'train_loss': [], 'train_acc': [], 'train_cnf': [],
    #         'val_loss': [], 'val_acc': [], 'val_cnf': [],
    #         'params': {
    #             'LR_0': LR_0,
    #             'GAMMA': GAMMA,
    #             'N_EPOCHS': N_EPOCHS,
    #             'STEP_SIZE': STEP_SIZE,
    #             'MOMENTUM': MOMENTUM,
    #             'WD': WD,
    #             'CLASS_WEIGHTS': CLASS_WEIGHTS
    #              }
    #        }

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            preds_acc = []
            labels_acc = []

            # Iterate over data.
            for i, batch in enumerate(dataloaders[phase]):
                if i % 100 == 0:
                    print('batch ' + str(i))
                inputs = batch[0]
                labels = batch[1]
                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                preds_acc.append(preds)
                labels_acc.append(labels)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc, cnf_mat = metrics(torch.cat(labels_acc).cpu(), torch.cat(preds_acc).cpu())
            
            data[phase + '_acc'].append(epoch_acc)
            data[phase + '_loss'].append(epoch_loss)
            data[phase + '_cnf'].append(cnf_mat)
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            print(cnf_mat)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, data

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #some arcane cuda magic on my pc and this breaks

if COLAB:
    device = torch.device("cuda:0")
    print('running on gpu! cuda:0')

else:
    device = torch.device("cpu")
    print('running on cpu!')

if SPLIT_MODE == 'manual': n_runs = 1 # if using a premade split, only run once
else: n_runs = N_SPLITS

all_train_data = []
for run in range(n_runs):
    trainloader, validloader = init_loaders()
    print(f'Run {run+1} out of {n_runs}')
    dataloaders = {'train': trainloader, 'val': validloader}
    dataset_sizes = {x: len(dataloaders[x])*BS for x in ['train', 'val']}

    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)

    model_ft = model_ft.to(device)
    
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=LR_0, momentum=MOMENTUM, weight_decay = WD)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=STEP_SIZE, gamma=GAMMA)
    model_ft, train_data = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                        num_epochs=N_EPOCHS)

    all_train_data.append(train_data)

save_data(all_train_data)