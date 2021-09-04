from torch._C import dtype
from torch.utils.data import dataset
from model.model import Model
from dataset import FVCDataset
import torch.nn as nn
import torch
import torch.utils.data as data
from torch import optim
from torch import Tensor
# import pytorch_ssim

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
        batch_size=16,
        shuffle=True,
    )

    # Optimizer用於調整model參數以減少model誤差的過程。Optimization algorithms定義了這個過程是如何執行的，
    # 常見的Optimizer如：SGD, ADAM, RMSProp。
    optimizer: optim.RMSprop = optim.RMSprop(model.parameters(), lr=0.00001, weight_decay=1e-8, momentum=0.9)

    # Loss function用於衡量得到結果與目標值之間的不相似程度，是我們在訓練過程中想要最小化的損失函數。
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