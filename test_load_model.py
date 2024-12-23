import pickle
import os

# Check if the model file exists
if not os.path.exists('model.pkl'):
    print("Model file does not exist.")
else:
    print("Model file found. Attempting to load...")

    try:
        with open('model.pkl', 'rb') as file:
            model = pickle.load(file)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        import pickle

# Load scaler.pkl
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Now you can use it
data = [[12, 45, 78]]  # Example input data
scaled_data = scaler.transform(data)
print("Scaled Data:", scaled_data)
