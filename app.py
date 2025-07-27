from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import joblib
import numpy as np
import torch
import os
app = FastAPI()

# เปิด CORS สำหรับ React (ปรับ domain ตามต้องการ)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ปรับตาม front-end URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
    
# โหลดโมเดล, label encoder, และ scaler สำหรับ MLP
model_mlp = tf.keras.models.load_model("modelMLP/model.h5")
label_encoder_mlp = joblib.load("modelMLP/label_encoder.pkl")
scaler_mlp = joblib.load("modelMLP/scaler.pkl")

# โหลดโมเดล, label encoder, และ scaler สำหรับ LSTM
model_lstm = tf.keras.models.load_model("modelLSTM/model_lstm.h5")
label_encoder_lstm = joblib.load("modelLSTM/label_encoder.pkl")
scaler_lstm = joblib.load("modelLSTM/scaler.pkl")

@app.get('/')
def get():
    return {"message": "Hello World V2"}

@app.websocket("/ws/mlp")
async def websocket_mlp(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            features = data.get("features", None)

            if features is None:
                await websocket.send_json({
                    "error": "Missing features",
                    "message": "ต้องส่งข้อมูลในรูปแบบ {'features': [...]}",
                })
                continue

            if not isinstance(features, list):
                await websocket.send_json({
                    "error": "Invalid features format",
                    "message": "features ต้องเป็น array ของตัวเลข",
                })
                continue

            expected_features = model_mlp.input_shape[1]
            if len(features) != expected_features:
                await websocket.send_json({
                    "error": "Invalid features length",
                    "message": f"ต้องส่ง features จำนวน {expected_features} ค่า (ได้ {len(features)} ค่า)",
                })
                continue

            if not all(isinstance(x, (int, float)) for x in features):
                await websocket.send_json({
                    "error": "Invalid feature values",
                    "message": "ทุกค่าใน features ต้องเป็นตัวเลข",
                })
                continue

            try:
                input_data = np.array(features).reshape(1, -1)
                input_data_scaled = scaler_mlp.transform(input_data)
                preds = model_mlp.predict(input_data_scaled)
                class_idx = np.argmax(preds, axis=1)[0]
                class_label = label_encoder_mlp.inverse_transform([class_idx])[0]

                await websocket.send_json({
                    "predicted_class": class_label,
                    "confidence": float(np.max(preds)),
                    "status": "success"
                })

            except Exception as e:
                await websocket.send_json({
                    "error": "Prediction failed",
                    "message": str(e),
                    "status": "error"
                })

    except WebSocketDisconnect:
        print("MLP Client disconnected")
    except Exception as e:
        await websocket.send_json({
            "error": "Server error",
            "message": str(e)
        })

@app.websocket("/ws/lstm")
async def websocket_lstm(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            features = data.get("features", None)

            if features is None:
                await websocket.send_json({
                    "error": "Missing features",
                    "message": "ต้องส่งข้อมูลในรูปแบบ {'features': [...]}",
                })
                continue

            if not isinstance(features, list):
                await websocket.send_json({
                    "error": "Invalid features format",
                    "message": "features ต้องเป็น array ของตัวเลข",
                })
                continue

            # สำหรับ LSTM input shape คือ (batch, timesteps, features)
            input_shape = model_lstm.input_shape  # (None, sequence_length, features_per_frame)
            sequence_length = input_shape[1]
            features_per_frame = input_shape[2]

            expected_length = sequence_length * features_per_frame

            if len(features) != expected_length:
                await websocket.send_json({
                    "error": "Invalid features length",
                    "message": f"ต้องส่ง features จำนวน {expected_length} ค่า (ได้ {len(features)} ค่า)",
                })
                continue

            if not all(isinstance(x, (int, float)) for x in features):
                await websocket.send_json({
                    "error": "Invalid feature values",
                    "message": "ทุกค่าใน features ต้องเป็นตัวเลข",
                })
                continue

            try:
                # แปลง features เป็น array shape (1, sequence_length, features_per_frame)
                input_data = np.array(features).reshape(1, sequence_length, features_per_frame)

                # scaler ต้องรับข้อมูล 2D shape (samples*timesteps, features)
                input_2d = input_data.reshape(-1, features_per_frame)
                input_scaled_2d = scaler_lstm.transform(input_2d)
                input_scaled = input_scaled_2d.reshape(1, sequence_length, features_per_frame)

                preds = model_lstm.predict(input_scaled)
                class_idx = np.argmax(preds, axis=1)[0]
                class_label = label_encoder_lstm.inverse_transform([class_idx])[0]

                await websocket.send_json({
                    "predicted_class": class_label,
                    "confidence": float(np.max(preds)),
                    "status": "success"
                })

            except Exception as e:
                await websocket.send_json({
                    "error": "Prediction failed",
                    "message": str(e),
                    "status": "error"
                })

    except WebSocketDisconnect:
        print("LSTM Client disconnected")
    except Exception as e:
        await websocket.send_json({
            "error": "Server error",
            "message": str(e)
        })

class_names = np.load("modelPT/classes.npy", allow_pickle=True)

class SimpleNN(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 128)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(128, 64)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

input_size = 171
num_classes = len(class_names)

model_pt = SimpleNN(input_size, num_classes)
model_pt.load_state_dict(torch.load("modelPT/landmark_model.pth", map_location=torch.device('cpu')))
model_pt.eval()

@app.websocket("/ws/pytorch")
async def websocket_pytorch(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            features = data.get("features", None)

            if features is None:
                await websocket.send_json({
                    "error": "Missing features",
                    "message": "ต้องส่งข้อมูลในรูปแบบ {'features': [...]}",
                })
                continue

            if not isinstance(features, list):
                await websocket.send_json({
                    "error": "Invalid features format",
                    "message": "features ต้องเป็น array ของตัวเลข",
                })
                continue

            if len(features) != input_size:
                await websocket.send_json({
                    "error": "Invalid features length",
                    "message": f"ต้องส่ง features จำนวน {input_size} ค่า (ได้ {len(features)} ค่า)",
                })
                continue

            if not all(isinstance(x, (int, float)) for x in features):
                await websocket.send_json({
                    "error": "Invalid feature values",
                    "message": "ทุกค่าใน features ต้องเป็นตัวเลข",
                })
                continue

            try:
                x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # shape: (1, 171)
                with torch.no_grad():
                    outputs = model_pt(x)
                    pred_idx = torch.argmax(outputs, dim=1).item()
                    pred_label = class_names[pred_idx]

                await websocket.send_json({
                    "predicted_class": pred_label,
                    "status": "success"
                })

            except Exception as e:
                await websocket.send_json({
                    "error": "Prediction failed",
                    "message": str(e),
                    "status": "error"
                })

    except WebSocketDisconnect:
        print("PyTorch Client disconnected")
    except Exception as e:
        await websocket.send_json({
            "error": "Server error",
            "message": str(e)
        })

model_path = "modelCNNKeras/gesture_cnn_model.h5"
label_encoder_path = "modelCNNKeras/label_encoder.pkl"

if not os.path.exists(model_path) or not os.path.exists(label_encoder_path):
    raise RuntimeError("Model หรือ Label encoder ไม่พบในโฟลเดอร์ modelCNNKeras")

model_cnn = tf.keras.models.load_model(model_path)
label_encoder = joblib.load(label_encoder_path)

@app.websocket("/ws/cnn")
async def websocket_cnn(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            features = data.get("features", None)
            if features is None:
                await websocket.send_json({
                    "error": "Missing features",
                    "message": "ต้องส่งข้อมูลในรูปแบบ {'features': [...]}",
                })
                continue
            
            if not isinstance(features, list) or not all(isinstance(x, (int, float)) for x in features):
                await websocket.send_json({
                    "error": "Invalid features format",
                    "message": "features ต้องเป็น array ของตัวเลข",
                })
                continue
            
            input_shape = model_cnn.input_shape  # เช่น (None, 128, 128, 1)
            expected_size = np.prod(input_shape[1:])
            if len(features) != expected_size:
                await websocket.send_json({
                    "error": "Invalid features length",
                    "message": f"ต้องส่ง features จำนวน {expected_size} ค่า (ได้ {len(features)} ค่า)",
                })
                continue

            try:
                input_array = np.array(features, dtype=np.float32).reshape((1,) + input_shape[1:])
                # Normalize pixel values แบบเดียวกับตอนเทรน
                input_array /= 255.0

                preds = model_cnn.predict(input_array)
                class_idx = np.argmax(preds, axis=1)[0]

                # เช็คชนิดของ label_encoder ว่าเป็น dict หรือใช้ inverse_transform()
                if isinstance(label_encoder, dict):
                    class_label = label_encoder.get(class_idx, "Unknown")
                else:
                    class_label = label_encoder.inverse_transform([class_idx])[0]

                await websocket.send_json({
                    "predicted_class": class_label,
                    "confidence": float(np.max(preds)),
                    "status": "success"
                })

            except Exception as e:
                await websocket.send_json({
                    "error": "Prediction failed",
                    "message": str(e),
                    "status": "error"
                })

    except WebSocketDisconnect:
        print("CNN Client disconnected")
    except Exception as e:
        await websocket.send_json({
            "error": "Server error",
            "message": str(e)
        })