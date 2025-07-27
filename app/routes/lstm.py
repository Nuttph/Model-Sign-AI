from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from tensorflow.keras.models import load_model
import joblib
import numpy as np

router = APIRouter()

# โหลดโมเดลและ scaler/label encoder
model_lstm = load_model("modelLSTM/model_lstm.h5")
label_encoder_lstm = joblib.load("modelLSTM/label_encoder.pkl")
scaler_lstm = joblib.load("modelLSTM/scaler.pkl")

@router.websocket("/ws/lstm")
async def websocket_lstm(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            features = data.get("features")

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

            input_shape = model_lstm.input_shape  # (None, seq_len, feat_per_frame)
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
                # reshape input (1, seq_len, feat_per_frame)
                input_data = np.array(features).reshape(1, sequence_length, features_per_frame)

                # scale ต้องแปลงเป็น 2D ก่อน (samples*timesteps, features)
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
