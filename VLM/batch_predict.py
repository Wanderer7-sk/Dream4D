import os
import torch
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
import csv

# 类别顺序要与你训练时保持一致
class_names = ['zoom_in', 'zoom_out', 'turn_left', 'turn_right', 'orbit', 'look_up', 'look_down', 'stationary']

# 图像预处理
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_model(model_path='camera_trajectory_resnet18.pth'):
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(model.fc.in_features, len(class_names))
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def draw_prediction(image, label, score):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    text = f"{label} ({score:.2f})"
    draw.rectangle([(0, 0), (image.width, 20)], fill=(255, 255, 255))
    draw.text((5, 2), text, fill=(0, 0, 0), font=font)
    return image

def batch_predict(image_dir, output_csv='predictions.csv', save_dir='labeled_images'):
    model = load_model()
    results = []
    os.makedirs(save_dir, exist_ok=True)

    for fname in sorted(os.listdir(image_dir)):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img_path = os.path.join(image_dir, fname)
        try:
            image = Image.open(img_path).convert('RGB')
            input_tensor = preprocess(image).unsqueeze(0)

            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)[0]
                pred_idx = torch.argmax(probs).item()
                label = class_names[pred_idx]
                confidence = probs[pred_idx].item()
                results.append((fname, label, f"{confidence:.4f}"))

                # 可视化 + 保存
                labeled_img = draw_prediction(image.copy(), label, confidence)
                labeled_img.save(os.path.join(save_dir, fname))

                print(f"{fname}: {label} ({confidence:.2f})")

        except Exception as e:
            print(f"Failed to process {fname}: {e}")

    # 写入 CSV 文件
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Filename', 'Predicted Class', 'Confidence'])
        writer.writerows(results)

    print(f"\n✅ Results saved to {output_csv}")
    print(f"✅ Labeled images saved to {save_dir}/")

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python batch_predict.py <image_folder>")
    else:
        batch_predict(sys.argv[1])
