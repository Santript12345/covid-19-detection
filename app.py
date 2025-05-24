import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
from torchvision.datasets import ImageFolder
# Define your ANN model
class ANN(nn.Module):
    def __init__(self, hidden_layer=64):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(120 * 120 * 3, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, 3)
        self.relu = nn.ReLU()

    def forward(self, img):
        out = img.view(-1, 120 * 120 * 3)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Function to load the model
def load_model(model_path="project_model.pth"):
    model = ANN(hidden_layer=64)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Define paths and transformations
data_path_test = "static/test/"
data_path_train = "static/train/"
img_size = 120
img_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load ImageFolder datasets
train_data = ImageFolder(root=data_path_train, transform=img_transform)
test_data = ImageFolder(root=data_path_test, transform=img_transform)
classes = train_data.classes

# Function to predict image class
def predict_img(img_path, model):
    img = Image.open(img_path)
    img = img_transform(img)  # Apply transformations
    img = img.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(img)
    _, predicted = torch.max(output, 1)
    predicted_class = classes[predicted.item()]
    return predicted_class

# Initialize Flask application
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Load the model
model = load_model()

@app.route('/')
def index():
    return render_template('index.html')



@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        prediction = predict_img(filepath, model)
        # Construct image_url relative to the 'static' folder
        image_url = f'uploads/{file.filename}'  # Adjust path based on your actual structure
        return render_template('result.html', prediction=prediction, image_url=image_url)



if __name__ == '__main__':
    app.run(debug=True)
