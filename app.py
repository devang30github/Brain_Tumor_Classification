import os
import uuid
from flask import Flask, request, jsonify, render_template, session
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn as nn

app = Flask(__name__)

# Set the secret key to enable session handling
app.secret_key = '7979u9'  # Replace with a secure key

# Define the path to save uploaded images
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG16, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VGG16(num_classes=4)  # Adjust num_classes based on your dataset
model.load_state_dict(torch.load('model/best_brain_tumor_vgg16.pth', map_location=device))
model.eval()
model.to(device)

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)

        if 'temp_files' not in session:
            session['temp_files'] = []
        session['temp_files'].append(filepath)

        # Process the image and make prediction
        grayscale_image = Image.open(filepath).convert('L')
        rgb_image = Image.merge("RGB", (grayscale_image, grayscale_image, grayscale_image))
        rgb_image = transform(rgb_image).unsqueeze(0).to(device)  # Add batch dimension

        with torch.no_grad():
            classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
            output = model(rgb_image)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            confidences = {classes[i]: float(probabilities[i]) for i in range(len(classes))}
            _, predicted = torch.max(output, 1)
            result = classes[predicted.item()]

        return jsonify({'result': result,'confidences': confidences})

@app.teardown_request
def cleanup(exception=None):
    temp_files = session.pop('temp_files', [])
    for temp_file in temp_files:
        try:
            os.remove(temp_file)
        except OSError:
            pass

if __name__ == '__main__':
    app.run(debug=True)
