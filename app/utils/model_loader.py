import joblib
import numpy as np
import tensorflow as tf
import torch
import pickle
from keras.models import load_model
from app.config import *

def load_mlp_components():
    model = load_model(MLP_MODEL_PATH)
    encoder = joblib.load(MLP_ENCODER_PATH)
    scaler = joblib.load(MLP_SCALER_PATH)
    return model, encoder, scaler

def load_lstm_components():
    model = load_model(LSTM_MODEL_PATH)
    encoder = joblib.load(LSTM_ENCODER_PATH)
    scaler = joblib.load(LSTM_SCALER_PATH)
    return model, encoder, scaler

def load_cnn_components():
    model = load_model(CNN_MODEL_PATH)
    with open(CNN_ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)
    return model, encoder

def load_torch_model():
    model = torch.load(TORCH_MODEL_PATH, map_location=torch.device('cpu'))
    model.eval()
    classes = np.load(TORCH_CLASSES_PATH)
    return model, classes
