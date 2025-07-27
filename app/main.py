from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import lstm
from app.config import ALLOWED_ORIGINS

app = FastAPI()

# CORS

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
@app.get("/")
def root():
    return {"message": "Hello World"}

# รวม WebSocket routes
# app.include_router(mlp.router)
app.include_router(lstm.router)
# app.include_router(pytorch.router)
# app.include_router(cnn.router)
