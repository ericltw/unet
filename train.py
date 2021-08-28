from torch._C import dtype
from torch.utils.data import dataset
from model.model import Model
from dataset import FVCDataset
import torch.nn as nn
import torch
import torch.utils.data as data
from torch import optim
from torch import Tensor

if __name__ == "__main__":
    # cpu or gpu
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # model
    model: Model = Model()
    model.to(device=device)
    
    # dataloader
    data_set = FVCDataset()
    train_loader = data.DataLoader(
        dataset=data_set,
        batch_size=1,
        shuffle=True,
    )

    # TODO: Optimizer
    optimizer: optim.RMSprop = optim.RMSprop(model.parameters(), lr=0.00001, weight_decay=1e-8, momentum=0.9)

    # TODO: Loss算法
    criterion: nn.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
    # best_loss統計，初始化為正無窮
    best_loss = float('inf')

    # TODO: 訓練epochs次
    epochs = 40
    for epoch in range(epochs):
        model.train()
        for image, label in train_loader:
            optimizer.zero_grad()

            image: Tensor = image.to(device=device, dtype=torch.float32)
            label: Tensor = label.to(device=device, dtype=torch.float32)
            pred: Tensor = model(image)

            # 計算loss
            loss = criterion(pred, label)
            print('Loss/train', loss.item())

            # 保存參數
            if loss < best_loss:
                best_loss = loss
                torch.save(model.state_dict(), 'best_model.pth')

            # 更新参数
            loss.backward()
            optimizer.step()