import os
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from torch.optim.lr_scheduler import StepLR
import numpy as np
import re
import matplotlib.pyplot as plt
from copy import deepcopy

# -------------- Simplicity is Elegancy -------------------




def plot_frame(frame):
        plt.figure(figsize=(10, 10))
        plt.imshow(frame.cpu().numpy())
        plt.axis('off')

def plot_video_frames(video_tensor, frame_n=0, name='frame'):
    # video_tensor: [channels, frames, height, width]
    frame = video_tensor[:,frame_n]
    frame = frame.permute(1, 2, 0).cpu().numpy()  # Change the order to [height, width, channels]
    plt.imshow(frame)
    plt.axis('off')
    #plt.savefig(str(frame_n)+'_'+name+'.png', dpi=300)
    plt.show()


def evaluate_model(model, data_loader):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')

            outputs = model(inputs) 

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(outputs[:,0].cpu().numpy())
            preds = (torch.sigmoid(outputs)+0.5).int()[:,0]
            all_preds.extend(preds.cpu().numpy())

    return all_labels,  all_probs, all_preds


def Scores (all_probs, all_preds, all_labels):
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    
    auc = roc_auc_score(all_labels, all_probs)

    conf_matrix = confusion_matrix(all_labels, all_preds)
    TN, FP, FN, TP = conf_matrix.ravel()

    sen = TP / (TP + FN)
    spe = TN / (TN + FP)

    gmean = np.sqrt(sen*spe)

    return acc, f1, precision, recall, sen, spe, gmean, auc, TN, FP, FN, TP


def sort_frames(file):
    # Extract the number from the filename using a regular expression
    numbers = re.findall(r'\d+', file)
    # Convert to integer and return for sorting
    return int(numbers[0]) if numbers else 0


class DFDCDataset(Dataset):
    def __init__(self, root_dir, split='TRAIN', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.data = []
        self._load_data()

    def _load_data(self):
        for label in ['FAKE', 'REAL']:
            label_path = os.path.join(self.root_dir, label, self.split)
            for id in os.listdir(label_path):
                id_path = os.path.join(label_path, id)
                frames = sorted(os.listdir(id_path), key=sort_frames)
                frames_with_id = [(id, frame) for frame in frames]
                self.data.append((frames_with_id, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frames_with_id, label = self.data[idx]
        images = []
        for id, frame in frames_with_id:
            image_path = os.path.join(self.root_dir, label, self.split, id, frame)
            image = Image.open(image_path)
            if self.transform:
                image = self.transform(image)
            images.append(image)
        images = torch.stack(images)
        images = images.permute(1, 0, 2, 3)

        label = 1 if label == 'FAKE' else 0
        return images, label


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

root_data = 'DFDC' 

# <---- this is one fold, use different folds to average the scores for a 10-cross validation
train_dataset = DFDCDataset(root_dir=root_data, split='TRAIN', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True)
val_dataset = DFDCDataset(root_dir=root_data, split='VAL', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=7, shuffle=False)
test_dataset = DFDCDataset(root_dir=root_data, split='TEST', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=7, shuffle=False)
# <----


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class ResNet3D_QDA(nn.Module):
    def __init__(self, layer_order, mode):
        super(ResNet3D_QDA, self).__init__()
        self.model = models.video.r3d_18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_ftrs//2)
        self.fc2 = nn.Linear(num_ftrs//2, 1)

        if mode == 'SU':
            for param in self.model.parameters():
                param.requires_grad = True
            for layer in list(self.model.children())[-3:]: 
                for param in layer.parameters():
                    param.requires_grad = True
            for param in self.model.fc.parameters():
                param.requires_grad = True

        if mode == 'GU':
            for param in self.model.parameters():
                param.requires_grad = False
            for layer in list(self.model.children())[layer_order:]: # -1: -2: -3:, ....  # <--- gradual unfreezing
                for param in layer.parameters():
                    param.requires_grad = True
            for param in self.model.fc.parameters():
                param.requires_grad = True

    def forward(self, x):
        x = QDA(x, 100) # <--- this is quantization differnce based process for enhancining deepfakes detection, 
        # its just the main reason of GU to smoothly adapt the pre-trained weigths to the new representation space
        x = self.model(x)
        x = self.fc2(x)
        return x

def QDA(Z, r):

    # plot_video_frames(Z[0])

    # <--- Quantization/Discretization
    Q = torch.round(Z * r) / r
    # plot_video_frames(Q[0])

    # <--- Differencing
    D = Z-Q
    # plot_video_frames(D[0])
    
    # <--- Aggregation of Cont and Disc
    ZA = D + Z
    # plot_video_frames(ZA[0])

    return ZA




criterion = nn.BCEWithLogitsLoss()
layer_order = 0
###lr_init_gu = 0.001
lr_init_su = 0.00005
total_params = count_trainable_parameters(deepcopy(ResNet3D_QDA(layer_order=None, mode='SU').to('cuda'))) # 33297857
print(total_params) 
best_score = 0.0
best_model_wts = None
torch.save(deepcopy(ResNet3D_QDA(layer_order=None, mode='SU').to('cuda').state_dict()), 'model_CGU.pth')
all_preds = []
all_labels = []


# CGU: It aims to smooth the weights anomalies and inbalancies between diferrent layers potentially caused during classic GU.
for cyrcle in range(3):
    # ------------------- SU ---------------------------------------------------------------------------------------------------------------
    # set model to Simultaneously Unfreezing (Unfreeze all)
    # set learning rate to be very small (e.g. 0.00005)

    model = ResNet3D_QDA(layer_order=None, mode='SU').to('cuda')
    model.load_state_dict(torch.load('model_CGU.pth'))
    optimizer = torch.optim.Adam(model.parameters(), lr = lr_init_su)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.7) # 

    num_epochs = 2
    for epoch in range(num_epochs):

        model.train()
        for i, (inputs, labels) in enumerate(train_loader):

            inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels.view(-1, 1).float())
            loss.backward()
            optimizer.step()

            preds = (torch.sigmoid(outputs)+0.5).int()[:,0]

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

        print(np.round(accuracy_score(all_labels, all_preds),4))
        scheduler.step()

        # Evaluate on validation set
        all_labels,   all_probs, all_preds  = evaluate_model(model, val_loader)
        acc, f1, precision, recall, sen, spe, gmean, auc, TN, FP, FN, TP = Scores (all_probs, all_preds, all_labels)
        print(f"Epoch {epoch+1}/{num_epochs} - Acc: {np.round(acc,4)}, F1: {np.round(f1,4)}, Precision: {np.round(precision,4)}, Sen: {np.round(sen,4)}, Spe: {np.round(spe,4)}, GMean: {np.round(gmean,4)}, Auc: {np.round(auc,4)}")

        score = gmean*auc
        print(score)
        if score > best_score:
            best_score = score
            best_model_wts = deepcopy(model.state_dict())
    torch.save(deepcopy(model.to('cuda').state_dict()), 'model_CGU.pth')


    # ------------------- GU ---------------------------------------------------------------------------------------------------------------
    layer_order += -1
    model = ResNet3D_QDA(layer_order=layer_order, mode='GU').to('cuda')
    model.load_state_dict(torch.load('model_CGU.pth'))

    current_trainable_params = count_trainable_parameters(model)
    ratio = current_trainable_params / total_params
    lr_init_gu = lr_init_su / np.sqrt(ratio) # Square Root Scaling 
    print(lr_init_gu) 

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_init_gu)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.7) # 

    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):

            inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels.view(-1, 1).float())
            loss.backward()
            optimizer.step()

            preds = (torch.sigmoid(outputs)+0.5).int()[:,0]

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
        print(np.round(accuracy_score(all_labels, all_preds),4))
        scheduler.step()

        # Evaluate on validation set
        all_labels,   all_probs, all_preds  = evaluate_model(model, val_loader)
        acc, f1, precision, recall, sen, spe, gmean, auc, TN, FP, FN, TP = Scores (all_probs, all_preds, all_labels)
        print(f"Epoch {epoch+1}/{num_epochs} - Acc: {np.round(acc,4)}, F1: {np.round(f1,4)}, Precision: {np.round(precision,4)}, Sen: {np.round(sen,4)}, Spe: {np.round(spe,4)}, GMean: {np.round(gmean,4)}, Auc: {np.round(auc,4)}")

        score = gmean*auc
        print(score)
        if score > best_score:
            best_score = score
            best_model_wts = deepcopy(model.state_dict())
    torch.save(deepcopy(model.to('cuda').state_dict()), 'model_CGU.pth')


# ------------------- Final - SU ---------------------------------------------------------------------------------------------------------------
model = ResNet3D_QDA(layer_order=None, mode='SU').to('cuda')
model.load_state_dict(torch.load('model_CGU.pth'))
optimizer = torch.optim.Adam(model.parameters(), lr=lr_init_su)
scheduler = StepLR(optimizer, step_size=5, gamma=0.7)
num_epochs = 10
for epoch in range(num_epochs):
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):

            inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels.view(-1, 1).float())
            loss.backward()
            optimizer.step()

            preds = (torch.sigmoid(outputs)+0.5).int()[:,0]

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
        print(np.round(accuracy_score(all_labels, all_preds),4))
        scheduler.step()

        # Evaluate on validation set
        all_labels,   all_probs, all_preds  = evaluate_model(model, val_loader)
        acc, f1, precision, recall, sen, spe, gmean, auc, TN, FP, FN, TP = Scores (all_probs, all_preds, all_labels)
        print(f"Epoch {epoch+1}/{num_epochs} - Acc: {np.round(acc,4)}, F1: {np.round(f1,4)}, Precision: {np.round(precision,4)}, Sen: {np.round(sen,4)}, Spe: {np.round(spe,4)}, GMean: {np.round(gmean,4)}, Auc: {np.round(auc,4)}")

        score = gmean*auc
        print(score)
        if score > best_score:
            best_score = score
            best_model_wts = deepcopy(model.state_dict())


# Save the best model weights
torch.save(best_model_wts, 'qda_CGA.pth')
model.load_state_dict(torch.load('qda_CGA.pth'))


# Evaluate on test set
all_labels,   all_probs, all_preds = evaluate_model(model, test_loader)
acc, f1, precision, recall, sen, spe, gmean, auc, TN, FP, FN, TP = Scores (all_probs, all_preds, all_labels)
print(f"Epoch {epoch+1}/{num_epochs} - Acc: {np.round(acc,4)}, F1: {np.round(f1,4)}, Precision: {np.round(precision,4)}, Sen: {np.round(sen,4)}, Spe: {np.round(spe,4)}, GMean: {np.round(gmean,4)}, Auc: {np.round(auc,4)}")




