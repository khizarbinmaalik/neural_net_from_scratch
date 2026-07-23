from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from model import load_model, predict
import uvicorn

app = FastAPI(title="MNIST Digit Classifier")

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (our HTML frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── LOAD MODEL ONCE AT STARTUP ────────────────────────────────────
params = load_model('mlp_weights.npz')
print("Model loaded successfully!")

# ── ROUTES ────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html") as f:
        return f.read()

@app.post("/predict")
async def predict_digit(file: UploadFile = File(...)):

    # Read image bytes from the uploaded file
    image_bytes = await file.read()

    # Run prediction
    result = predict(image_bytes, params)

    return JSONResponse(content=result)


# ── RUN ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
