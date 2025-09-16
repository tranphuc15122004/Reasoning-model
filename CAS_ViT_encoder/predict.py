import torch

from timm.models import create_model
import utils as utils

from model import *
from PIL import Image
import torchvision.transforms as transforms

# Đường dẫn tới file checkpoint
checkpoint_path = r"C:\\PHUC\\CAS-ViT_plantdoc\\output\\checkpoint-best_224.pth"

# Thiết bị (CPU hoặc GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Tải checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device ,weights_only= False)

# Tạo mô hình
model = create_model(
    'rcvit_m',
    pretrained=False,
    num_classes=27,  # Số lớp là 27
    drop_path_rate=0.1,
    layer_scale_init_value=1e-6,
    head_init_scale=1.0,
    input_res=256,  # Kích thước đầu vào là 256x256
    classifier_dropout=0.3,
    distillation=False,
)

# Load trạng thái của mô hình từ checkpoint
model.load_state_dict(checkpoint['model'])
model.to(device)  # Chuyển mô hình lên thiết bị
model.eval()  # Chuyển sang chế độ đánh giá

# Định nghĩa các transforms để tiền xử lý ảnh
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize về kích thước đầu vào của mô hình
    transforms.ToTensor(),  # Chuyển ảnh thành tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Chuẩn hóa
])

# Hàm dự đoán cho một ảnh
def predict_image(image_path, model, device, transform):
    """
    Nhận vào đường dẫn của ảnh, trả về lớp dự đoán và xác suất cao nhất.
    
    Args:
        image_path (str): Đường dẫn tới file ảnh.
        model: Mô hình đã huấn luyện.
        device: Thiết bị (CPU/GPU).
        transform: Các phép biến đổi để tiền xử lý ảnh.
    
    Returns:
        predicted_class (int): Lớp dự đoán.
        probabilities (torch.Tensor): Xác suất của tất cả các lớp.
    """
    # Mở và tiền xử lý ảnh
    image = Image.open(image_path).convert('RGB')  # Đảm bảo ảnh là RGB
    image = transform(image)  # Áp dụng transforms
    image = image.unsqueeze(0)  # Thêm batch dimension (1, C, H, W)
    image = image.to(device)  # Chuyển lên thiết bị

    # Không tính gradient (đánh giá)
    with torch.no_grad():
        outputs = model(image)  # Dự đoán từ mô hình
        probabilities = torch.softmax(outputs, dim=1)  # Chuyển thành xác suất
        predicted_class = torch.argmax(probabilities, dim=1).item()  # Lớp có xác suất cao nhất

    return predicted_class, probabilities[0].cpu().numpy()  # Trả về lớp và xác suất

def predict_image(image_path, model, device, transform):
    """
    Nhận vào đường dẫn của ảnh, trả về lớp dự đoán và xác suất cao nhất.
    
    Args:
        image_path (str): Đường dẫn tới file ảnh.
        model: Mô hình đã huấn luyện.
        device: Thiết bị (CPU/GPU).
        transform: Các phép biến đổi để tiền xử lý ảnh.
    
    Returns:
        predicted_class (int): Lớp dự đoán.
        probabilities (torch.Tensor): Xác suất của tất cả các lớp.
    """
    # Mở và tiền xử lý ảnh
    image = Image.open(image_path).convert('RGB')  # Đảm bảo ảnh là RGB
    image = transform(image)  # Áp dụng transforms
    image = image.unsqueeze(0)  # Thêm batch dimension (1, C, H, W)
    image = image.to(device)  # Chuyển lên thiết bị

    # Không tính gradient (đánh giá)
    with torch.no_grad():
        outputs = model(image)  # Dự đoán từ mô hình
        probabilities = torch.softmax(outputs, dim=1)  # Chuyển thành xác suất
        predicted_class = torch.argmax(probabilities, dim=1).item()  # Lớp có xác suất cao nhất

    return predicted_class, probabilities[0].cpu().numpy() , outputs  # Trả về lớp và xác suất


# Ví dụ sử dụng
if __name__ == "__main__":
    # Đường dẫn tới ảnh cần dự đoán
    image_path = r"C:\PHUC\CAS-ViT_plantdoc\PlantDoc-Dataset\train\Bell_pepper leaf\Bell_pepper leaf (1).jpg"  # Thay bằng đường dẫn thực tế của bạn
    print(image_path)
    
    try:
        # Gọi hàm dự đoán
        predicted_class, probabilities , outputs = predict_image(image_path, model, device, transform)

        # In kết quả
        print(f"Predicted class: {predicted_class}")  # In lớp dự đoán (0 đến 26, tùy thuộc vào số lớp)
        print("Probabilities for all classes:")
        for i, prob in enumerate(probabilities):
            print(f"Class {i}: {prob:.4f}")

    except Exception as e:
        print(f"Error during prediction: {e}")
    
    print(outputs)
