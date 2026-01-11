# Dog vs Cat Image Classifier 🐕🐈

A simple convolutional neural network (CNN) built with PyTorch that can tell if its a dog or a cat in image.

## 🎯 Results
- **Accuracy:** 80% on test data
- **Dataset:** 23,410 images from Microsoft Cats vs Dogs (Hugging Face)
- **Training time:** ~5 minutes on GPU

## 🚀 Quick Start

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run the Model
```bash
python train.py
```

The script will:
1. Automatically download the dataset from Hugging Face (first run only)
2. Train a CNN for 5 epochs
3. Display training progress and accuracy
4. Generate prediction visualizations
5. Save the trained model

## 📊 What It Does

This model takes a color image and predicts whether it contains a dog or a cat.

**Architecture:**
- 3 Convolutional layers (32, 64, 128 filters)
- Max pooling after each conv layer
- 2 Fully connected layers (256, 2 neurons)
- Dropout for regularization

**Input:** 128×128 RGB image  
**Output:** Dog or Cat prediction

## 📁 Output Files

After running, you'll get:
- `dog_cat_classifier.pth` - Trained model weights
- `dog_cat_predictions.png` - Sample predictions visualization

## 🛠️ Requirements

- Python 3.7+
- PyTorch
- torchvision
- Hugging Face datasets
- Pillow (PIL)
- matplotlib

## 📝 Usage Example

Classify your own images:
```python
import torch
from PIL import Image
from torchvision import transforms

# Load the trained model
model = DogCatClassifier()
model.load_state_dict(torch.load('dog_cat_classifier.pth'))
model.eval()

# Prepare your image
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

img = Image.open('my_pet.jpg').convert('RGB')
img_tensor = transform(img).unsqueeze(0)

# Make prediction
with torch.no_grad():
    output = model(img_tensor)
    prediction = output.argmax().item()
    confidence = torch.softmax(output, dim=1).max().item() * 100

result = 'Dog' if prediction == 1 else 'Cat'
print(f"This is a {result}! (Confidence: {confidence:.1f}%)")
```

## 🎓 Learning Journey

This is my second deep learning project! Key things I learned:
- Working with color images (RGB channels)
- Using Convolutional Neural Networks for image classification
- Loading datasets from Hugging Face
- Image preprocessing and normalization
- Building more complex architectures than my first MNIST project

## 🤖 Development Note

This project was developed with assistance from Claude for learning purposes.

## 📜 License

MIT License - Feel free to use this for learning!

## 🙏 Acknowledgments

- Microsoft Cats vs Dogs dataset (via Hugging Face)
- PyTorch and torchvision libraries
- Hugging Face datasets library
- Claude for code guidance and assistance

## ✨ Results
- Epoch 1/5
    ✅ Loss: 0.5743 | Train: 70.38% | Test: 52.09%
- Epoch 2/5
    ✅ Loss: 0.4600 | Train: 78.78% | Test: 75.78%
- Epoch 3/5
    ✅ Loss: 0.3884 | Train: 82.48% | Test: 64.97%
- Epoch 4/5
    ✅ Loss: 0.3172 | Train: 86.23% | Test: 74.52%
- Epoch 5/5
    ✅ Loss: 0.2505 | Train: 89.64% | Test: 78.47%
- ✨ Training complete! Final test accuracy: 78.47%
<img width="2174" height="886" alt="image" src="https://github.com/user-attachments/assets/4a8d1d0b-9c47-427c-b9d0-1aca77659e86" />

