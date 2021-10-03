import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.distributions import Normal
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from torch.optim import AdamW
import torch.nn.functional as F
from dataset import MnistAnomaly


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder,self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)  
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.t_conv = nn.ConvTranspose2d(512, 256, 3, stride=1)
        self.t_conv1 = nn.ConvTranspose2d(256, 128, 2, stride=1)
        self.t_conv2 = nn.ConvTranspose2d(128, 64, 2, stride=1)
        self.t_conv3 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.t_conv4 = nn.ConvTranspose2d(32, 1, 2, stride=2)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool(x)
        x = F.relu(self.t_conv(x))
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = F.relu(self.t_conv3(x))
        x = self.t_conv4(x)
        return x
    
def get_scores(autoencoder,x_test):
    with torch.no_grad():
        reconstructed_test = autoencoder(x_test.to(device))
        differences = x_test - reconstructed_test.cpu()
        diff_norm = torch.norm(differences,dim=[2,3]).view(-1)
    return F.sigmoid(-diff_norm)


def compress(x):
    return x.reshape(len(x), -1).sum(-1)


aucs = []
ordered = [1,9,7,2,3,4,5,6,8,0]
for i in ordered:
    abnormal_digit = [i]
    train_set = MnistAnomaly(
        root=".", train=True, transform=transforms.ToTensor(), anomaly_categories=abnormal_digit,download=True
    )

    test_set = MnistAnomaly(
        root=".", train=False, transform=transforms.ToTensor(), anomaly_categories=abnormal_digit,download=True
    )
    train_loader = DataLoader(train_set, batch_size=len(train_set))
    x, _ = next(iter(train_loader))
    x = compress(x)
    sd,m = x.std(),x.mean()
    device = "cuda" if torch.cuda.is_available() else "cpu"


    batch_size = 2048
    train_loader = DataLoader(train_set, batch_size=batch_size)

    autoencoder = AutoEncoder().to(device)
    optimizer = AdamW(params=autoencoder.parameters())
    criterion = nn.MSELoss()

    for _ in tqdm(range(100)):
        epoch = range(len(train_loader))
        for batch in epoch:
            batch_images, _ = next(iter(train_loader))
            reconstructed_images = autoencoder.forward(batch_images.to(device))
            loss = criterion(batch_images.to(device),reconstructed_images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # test model
    test_loader = DataLoader(test_set, batch_size=len(test_set))
    x_test, y_test = next(iter(test_loader))

    # compute score
    score_test = get_scores(autoencoder,x_test)

    # compute rocauc
    roc_auc = roc_auc_score(y_test, score_test)
    print('roc_auc======',roc_auc)
    aucs.append(roc_auc)
    
print("roc_auc per digit:")
print(["{:0.3f} ".format(auc) for auc in aucs])
print("average roc_auc:")
print("{:0.3f}".format(torch.tensor(aucs).mean()))

'''
RESULTS--------------------
roc_auc per digit:
['0.374 ', '0.564 ', '0.701 ', '0.929 ', '0.841 ', '0.785 ', '0.846 ', '0.922 ', '0.856 ', '0.863 ']
average roc_auc:
0.768
'''
