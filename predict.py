import glob
import numpy as np
import torch
import os
import cv2
from model.model import Model
from torchvision.io import read_image
from torch import Tensor

if __name__ == "__main__":
    # 選擇設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 載入unet
    net = Model()
    net.to(device=device)
    net.load_state_dict(torch.load('best_model.pth', map_location=device))
    net.eval()
    
    tests_path = glob.glob('data/test/*.png')
   
    for test_path in tests_path:
        # 保存结果地址
        save_res_path = test_path.split('.')[0] + '_res.png'
        # 讀取圖片
        img = cv2.imread(test_path)
        # 轉為灰階圖
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # 轉為batch為1，通道為1，大小為512*512
        img = img.reshape(1, 1, img.shape[0], img.shape[1])
        # 轉為tensor
        img_tensor = torch.from_numpy(img)
        # 將tensor copy到device中，只用cpu就是copy到cpu中，用cuda就是copy到cuda中。
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)
        # 預測
        pred = net(img_tensor)
        # 提取结果
        pred = np.array(pred.data.cpu()[0])[0]
        # 處理結果
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0
        # 保存圖片
        cv2.imwrite(save_res_path, pred)