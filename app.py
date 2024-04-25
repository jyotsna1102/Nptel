from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns 

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file:
        # Save the uploaded file
        file_path = 'uploads/' + file.filename
        file.save(file_path)

        # Process the data
        df = pd.read_csv(file_path)
        
        # Your existing code for data analysis and model training
        # For demonstration, I'm just returning a dummy accuracy and plot path
        accuracy = 85.3
        plot_path = 'static/plot.png'

        return jsonify({'accuracy': accuracy, 'plot_path': plot_path})
    else:
        return jsonify({'error': 'No file uploaded'})

if __name__ == '__main__':
    app.run(debug=True)
