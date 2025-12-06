import os
import io
import torch
from flask import Flask,request,render_template,jsonify
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0,EfficientNet_B0_Weights


app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER,exist_ok=True)

classes = ['Mild','Moderate','No_DR','Proliferate_DR','Severe']

model_path = './models/best_fine_tuned_model.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    model = efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_ftrs,len(classes))

    state_dict = torch.load(model_path,map_location=device)

    model.load_state_dict(state_dict)

    model.eval()
    print(f'Model loaded successfully from : {model_path}')
except FileNotFoundError:
    print(f'Error occured while loading the model at :- {model_path}')
    model = None


preprocess = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])


def predict_image(image_bytes):
    if model is None:
        return "Model not loaded" , -1

    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        input_batch = input_batch.to(device)
        output = model(input_batch)

        _,predicted_idx = torch.max(output,1)
        predicted_class = classes[predicted_idx.item()]

        return predicted_class,predicted_idx.item()


# ---------------- Flask routes ------------


@app.route("/")
def index():
    return render_template('index.html')



@app.route('/predict',methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error' : 'Please enter a file'}) , 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error' : 'Please enter a valid file'}) , 400

    if file:
        img_bytes = file.read()

        predicted_class, class_index = predict_image(img_bytes)

        return jsonify({
            "prediction" : predicted_class,
            "success" : True
        })


if __name__ == '__main__':
    if model is not None:
        app.run(debug=True)
    else:
        print('Flask server not found.')






