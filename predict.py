import glob
import numpy as np
import torch
import os
import cv2
from model.model import Model
from torchvision.io import read_image
from torch import Tensor
import torchvision.transforms as transforms

if __name__ == "__main__":
    # 選擇設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 載入unet
    net = Model()
    net.to(device=device)
    net.load_state_dict(torch.load('best_model.pth', map_location=device))
    net.eval()
    
    tests_path = glob.glob('data/test/*.png')
    resizeTo160 = transforms.Compose([
        transforms.Resize(160)
    ])
    resizeToBackTo150 = transforms.Compose([
        transforms.Resize(150)
    ])

    for test_path in tests_path:
        # 保存结果地址
        save_res_path = test_path.split('.')[0] + '_res.png'
        # 讀取圖片
        image = read_image(test_path)

        image = image.unsqueeze(1)

        image = resizeTo160(image) 
        # 將tensor copy到device中，只用cpu就是copy到cpu中，用cuda就是copy到cuda中。
        image = image.to(device=device, dtype=torch.float32)
        
        # 預測
        pred = net(image)
        
        # pred = resizeToBackTo150(pred)
        
        ####################
        # pred = pred / torch.max(pred)
        pred = pred / pred.max()
        pred = pred * 255
        print(pred)
        
        # # TODO: 提取结果
        pred = np.array(pred.data.cpu()[0])[0]
        # 保存圖片
        cv2.imwrite(save_res_path, pred)