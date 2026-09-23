import io
import time
import base64
import torch
import cv2
import numpy as np
import albumentations as A
from contextlib import asynccontextmanager
from albumentations.pytorch import ToTensorV2
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from src.model import SpatialMoESODNet

# Global variables for model and transforms
device = torch.device('cpu') # Forced to CPU to avoid CUDA out of memory
model = None
transform = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, transform
    print(f"Loading model on {device}...")
    try:
        checkpoint_path = "checkpoints/best_new_1.pth"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        model_cfg = checkpoint.get('config', {}).get('model', {})
        use_deep_supervision = model_cfg.get('deep_supervision', True)
        num_experts = model_cfg.get('num_experts', 6)
        top_k = model_cfg.get('top_k', 2)
        window_size = model_cfg.get('window_size', 8)
        
        model = SpatialMoESODNet(
            num_experts=num_experts,
            k=top_k,
            gate_mode=model_cfg.get('gate_mode', 'renormalized'),
            window_size=window_size,
            moe_type=model_cfg.get('moe_type', 'sparse'),
            use_deep_supervision=use_deep_supervision
        ).to(device)
        
        clean_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
        model.load_state_dict(clean_state_dict, strict=False)
        model.eval()
        
        transform = A.Compose([
            A.Resize(320, 320),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        print("Model loaded successfully!")
    except Exception as e:
        model = None
        print(f"Error loading model: {e}")
        print("Make sure best.pth is in the 'checkpoints' folder!")
    yield

app = FastAPI(title="Spatial-MoE SOD Inference", lifespan=lifespan)

# Enable CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None or transform is None:
        return JSONResponse({"error": "Model or transforms not loaded correctly"}, status_code=500)
        
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return JSONResponse({"error": "Invalid or unsupported image format."}, status_code=400)
        original_h, original_w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        start_time = time.time()
        input_tensor = transform(image=image_rgb)['image'].unsqueeze(0).to(device)
        input_tensor_flipped = torch.flip(input_tensor, dims=[3])
        
        # Inference (with TTA)
        with torch.no_grad():
            out, _ = model(input_tensor)
            out_flipped, _ = model(input_tensor_flipped)
            
            pred_sal = torch.sigmoid(out.saliency_logits)
            pred_sal_flipped = torch.sigmoid(out_flipped.saliency_logits)
            
            # Flip the flipped prediction back
            pred_sal_flipped_back = torch.flip(pred_sal_flipped, dims=[3])
            
            # Average the predictions
            pred_sal_avg = (pred_sal + pred_sal_flipped_back) / 2.0
            
            prob_mask = pred_sal_avg.squeeze().cpu().numpy()
        
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Resize mask back to original image size
        prob_mask_resized = cv2.resize(prob_mask, (original_w, original_h))
        binary_mask = (prob_mask_resized > 0.5).astype(np.uint8) * 255
        
        # Encode mask to base64 to send to frontend
        _, buffer = cv2.imencode('.png', binary_mask)
        mask_b64 = base64.b64encode(buffer).decode('utf-8')
        
        max_conf = float(np.max(prob_mask))
        
        return {
            "mask_b64": mask_b64,
            "inference_time_ms": round(inference_time, 2),
            "original_resolution": f"{original_w}x{original_h}",
            "model_resolution": "320x320",
            "max_confidence": round(max_conf, 4)
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# Serve static files last
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
