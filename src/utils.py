import os
import pickle


#save the model as pickle file
def save_model(model, path):

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as file:
        pickle.dump(model, file)
    print(f"Best model saved at {path}")

#Load the model  from a pickle file
def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model found at {path}")

    with open(path, 'rb') as file:
        model = pickle.load(file)

    print(f"Model loaded from {path}")
    return model