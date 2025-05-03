import torch
import torchvision.transforms as T
import torchvision
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
from matplotlib import rcParams

rcParams['font.family'] = 'Microsoft JhengHei'

# 設定 GPU / CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 載入預訓練模型
model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True).to(device)
model.eval()

# 下載圖片（街景）
def download_image(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("圖片下載失敗")
    image = Image.open(BytesIO(response.content)).convert("RGB")
    return np.array(image)

# 圖片前處理轉為 Tensor
def preprocess(image_np):
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize(520),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    return transform(image_np).unsqueeze(0).to(device)

# 將模型輸出轉為彩色 mask
def decode_segmentation(output):
    output_predictions = output.argmax(0).cpu().numpy()
    label_colors = np.array([
        (0, 0, 0),       # 0=background
        (128, 64,128),   # 1=road
        (244, 35,232),   # 2=sidewalk
        (70, 70, 70),    # 3=building
        (102,102,156),   # 4=wall
        (190,153,153),   # 5=fence
        (153,153,153),   # 6=pole
        (250,170, 30),   # 7=traffic light
        (220,220,  0),   # 8=traffic sign
        (107,142, 35),   # 9=vegetation
        (152,251,152),   # 10=terrain
        (70,130,180),    # 11=sky
        (220, 20, 60),   # 12=person
        (255,  0,  0),   # 13=rider
        (0,  0,142),     # 14=car
        (0,  0, 70),     # 15=truck
        (0, 60,100),     # 16=bus
        (0, 80,100),     # 17=train
        (0,  0,230),     # 18=motorcycle
        (119, 11, 32)    # 19=bicycle
    ])
    r = label_colors[output_predictions][:,:,0]
    g = label_colors[output_predictions][:,:,1]
    b = label_colors[output_predictions][:,:,2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb

# 疊加 segmentation mask 到原圖
def overlay(image_np, mask_np, alpha=0.6):
    mask_np = mask_np.astype(np.uint8)  # ✅ 確保型別正確
    return cv2.addWeighted(image_np, 1 - alpha, mask_np, alpha, 0)

# 主要處理函數
def process_image(image_np):
    input_tensor = preprocess(image_np)
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    mask = decode_segmentation(output)
    return overlay(cv2.resize(image_np, (mask.shape[1], mask.shape[0])), mask)

# 主程式
if __name__ == "__main__":
    url = "https://ultralytics.com/images/bus.jpg"  # 道路場景
    original = download_image(url)

    # 建立三種版本的圖片
    image_original = original.copy()
    image_blur = cv2.GaussianBlur(original, (11, 11), 0)
    image_bright = cv2.convertScaleAbs(original, alpha=1.3, beta=40)

    # 分別處理三張圖片
    result_original = process_image(image_original)
    result_blur = process_image(image_blur)
    result_bright = process_image(image_bright)

    # 顯示結果
    titles = ["原圖", "模糊處理", "亮度/對比調整"]
    images = [result_original, result_blur, result_bright]

    plt.figure(figsize=(18, 6))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.title(titles[i], fontsize=14)
        plt.axis('off')
    plt.suptitle("DeepLabV3 語意分割：不同前處理比較", fontsize=16)
    plt.tight_layout()
    plt.show()