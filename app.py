from flask import Flask, render_template, request
import pickle
import torch
from scripts.model import SpamClassifierV0

app = Flask(__name__)

vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

model_0 = SpamClassifierV0()
model_0.load_state_dict(torch.load('models/spam_classifier_v0.pth', weights_only=True))
model_0.eval()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        email_text = request.form['email']

        email_vector = vectorizer.transform([email_text])

        email_dense = email_vector.toarray()
        email_tensor = torch.tensor(email_dense, dtype=torch.float)

        with torch.no_grad():
            y_logits = model_0(email_tensor).squeeze()
            y_pred = torch.round(torch.sigmoid(y_logits))

        result = "Spam" if y_pred.item() == 1 else "Not Spam"

        return render_template('result.html', prediction=result, email=email_text)


if __name__ == '__main__':
    app.run(debug=False)
