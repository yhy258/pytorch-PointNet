import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from modules import PointNet_Classification
from losses import get_classification_loss
from augmentation import *
from dataset import PointCloudData

from Config import Config

def train(model, optim, save_path, train_loader, criterion, val_loader=None,  epochs=1, notebook = False):
    if notebook:
        from tqdm.notebook import tqdm
    else :
        from tqdm import tqdm

    for epoch in range(epochs):
        pointnet.train()
        running_loss = 0.0
        for i, data in enumerate(tqdm(train_loader), 0):
            inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)

            outputs, input_matrix, feature_matrix = pointnet(inputs.transpose(1 ,2))

            loss = criterion(outputs, labels, feature_matrix)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 5 == 4:    # print every 5 mini-batches
                print('[Epoch: %d, Batch: %4d / %4d], loss: %.3f' %
                      (epoch + 1, i + 1, len(train_loader), running_loss / 5))
                running_loss = 0.0

        pointnet.eval()
        correct = total = 0

        # validation
        if val_loader:
            with torch.no_grad():
                for data in val_loader:
                    inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
                    outputs, __, __ = pointnet(inputs.transpose(1 ,2))
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            val_acc = 100. * correct / total
            print('Valid accuracy: %d %%' % val_acc)

        # save the model
        torch.save(pointnet.state_dict(), "save.pth")

if __name__ == "__main__":
    cfg = Config()

    path = cfg.path # Data path. (ModelNet Dataset 40 기준.)
    save_path = cfg.save_path

    train_transforms = transforms.Compose([
        PointSampler(cfg.point_sampling_num),
        Normalize(),
        RandRotation_z(),
        RandomNoise(),
        ToTensor()
    ])

    train_ds = PointCloudData(path, transform=train_transforms)
    valid_ds = PointCloudData(path, valid=True, folder='test', transform=train_transforms)

    train_loader = DataLoader(dataset=train_ds, batch_size=cfg.batchsize, shuffle=True)
    valid_loader = DataLoader(dataset=valid_ds, batch_size=cfg.valid_batchsize)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    pointnet = PointNet_Classification(cfg.class_num)
    pointnet.to(device)

    optimizer = torch.optim.Adam(pointnet.parameters(), lr=cfg.lr)

    criterion = get_classification_loss()
    train(pointnet, optimizer, save_path, train_loader, criterion, valid_loader, epochs = cfg.epochs, notebook = cfg.notebook)