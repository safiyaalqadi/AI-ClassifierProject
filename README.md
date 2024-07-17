To Make It Work Please Read This File:

Setup
Create a virtual environment: bash python -m venv myenv 

Activate the virtual environment: - On Windows: bash myenv\Scripts\activate  - On macOS/Linux: bash source myenv/bin/activate 

Install the required packages: bash pip install -r requirements.txt 

Usage
Training the Model
To train the model, run the ai.py script. This script will load the Oxford Flowers 102 dataset, create training, validation, and test sets, build and train the model, and save the trained model.

Example Command
python predict.py test_images/hard-leaved_pocket_orchid.jpg --top_k 5 --category_names label_map.json

Example Output
Predictions for the image: 1: hard-leaved pocket orchid with probability 0.9971 2: bearded iris with probability 0.0019 3: canterbury bells with probability 0.0002 4: anthurium with probability 0.0001 5: moon orchid with probability 0.0001
