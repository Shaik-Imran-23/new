import os
import json
import cv2
import numpy as np
from pdf2image import convert_from_path
from ultralytics import YOLO
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

# ===============================
# CONFIG
# ===============================
PDF_PATH = "input/DD800488-3-00.pdf"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

YOLO_MODEL_PATH = "../training_model/runs/detect/ga_find_circle/weights/best.pt"

# Performance settings
BATCH_SIZE = 8  # Process multiple circles at once
USE_GPU = torch.cuda.is_available()
DEVICE = "cuda" if USE_GPU else "cpu"

print(f"🖥️  Device: {DEVICE}")
print(f"📦 Batch size: {BATCH_SIZE}")

# ===============================
# LOAD MODELS (ONCE!)
# ===============================
print("🔄 Loading YOLO model...")
yolo = YOLO(YOLO_MODEL_PATH)
print("✅ YOLO loaded!")

print(f"🔄 Loading Moondream model on {DEVICE}...")
model_id = "vikhyatk/moondream2"

# Load with optimal settings
if USE_GPU:
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        trust_remote_code=True,
        revision="2025-01-09",
        torch_dtype=torch.float16,  # Use FP16 on GPU
        device_map="auto"
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        trust_remote_code=True,
        revision="2025-01-09",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    model = model.to(DEVICE)

tokenizer = AutoTokenizer.from_pretrained(model_id, revision="2025-01-09")
print("✅ Moondream loaded!")

# ===============================
# OPTIMIZED PREDICTION FUNCTION
# ===============================
def predict_number_batch(crop_images):
    """Process multiple crops at once for speed"""
    results = []
    
    for crop_img in crop_images:
        try:
            pil_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
            
            # Optimized prompt
            prompt = "Number in circle:"
            
            # Encode and predict
            enc_image = model.encode_image(pil_img)
            answer = model.answer_question(enc_image, prompt, tokenizer)
            
            # Extract number
            number = int(''.join(filter(str.isdigit, answer)))
            results.append(number)
        except Exception as e:
            print(f"      ⚠️  Error: {e}")
            results.append(None)
    
    return results

def predict_number_single(crop_img):
    """Single prediction with error handling"""
    try:
        pil_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
        prompt = "Number in circle:"
        
        enc_image = model.encode_image(pil_img)
        answer = model.answer_question(enc_image, prompt, tokenizer)
        
        number = int(''.join(filter(str.isdigit, answer)))
        return number
    except:
        return None

# ===============================
# PDF → IMAGES
# ===============================
print(f"🔄 Converting PDF to images...")
pages = convert_from_path(PDF_PATH, dpi=300)
print(f"✅ Converted {len(pages)} pages")

results = []
total_circles = 0
total_vlm_time = 0

# ===============================
# MAIN PIPELINE (OPTIMIZED)
# ===============================
print("\n🔄 Processing pages...")

for page_no, page in enumerate(pages, start=1):
    print(f"\n📄 Page {page_no}/{len(pages)}")
    
    page_img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
    vis_img = page_img.copy()

    # YOLO detection
    yolo_start = time.time()
    detections = yolo(page_img)[0]
    yolo_time = time.time() - yolo_start
    
    num_circles = len(detections.boxes)
    print(f"   🎯 Found {num_circles} circles (YOLO: {yolo_time:.2f}s)")
    
    if num_circles == 0:
        continue

    # Collect all crops
    crops = []
    boxes_data = []
    
    for box in detections.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = page_img[y1:y2, x1:x2]
        
        if crop.size == 0:
            continue
            
        crops.append(crop)
        boxes_data.append((x1, y1, x2, y2))
    
    # Process in batches for speed
    print(f"   🔍 VLM Processing {len(crops)} circles...")
    vlm_start = time.time()
    
    if BATCH_SIZE > 1:
        # Batch processing
        all_predictions = []
        for i in range(0, len(crops), BATCH_SIZE):
            batch = crops[i:i+BATCH_SIZE]
            batch_predictions = predict_number_batch(batch)
            all_predictions.extend(batch_predictions)
            print(f"      Batch {i//BATCH_SIZE + 1}/{(len(crops)-1)//BATCH_SIZE + 1} done")
    else:
        # Single processing
        all_predictions = [predict_number_single(crop) for crop in crops]
    
    vlm_time = time.time() - vlm_start
    total_vlm_time += vlm_time
    total_circles += len(crops)
    
    avg_time_per_circle = vlm_time / len(crops) if crops else 0
    print(f"   ✅ VLM done in {vlm_time:.2f}s ({avg_time_per_circle:.2f}s per circle)")
    
    # Draw and save results
    for idx, (balloon_number, (x1, y1, x2, y2)) in enumerate(zip(all_predictions, boxes_data), 1):
        if balloon_number is None:
            continue
        
        # Draw on image
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            vis_img,
            str(balloon_number),
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )
        
        # Save result
        results.append({
            "page": page_no,
            "balloon_number": balloon_number,
            "bbox": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            }
        })
    
    # Save page image
    out_img_path = os.path.join(OUTPUT_DIR, f"page_{page_no}_detections.jpg")
    cv2.imwrite(out_img_path, vis_img)
    print(f"   💾 Saved: {out_img_path}")

# ===============================
# SAVE JSON & STATS
# ===============================
out_json = os.path.join(OUTPUT_DIR, "balloon_results.json")
with open(out_json, "w") as f:
    json.dump(results, f, indent=4)

print(f"\n{'='*50}")
print(f"✅ PROCESSING COMPLETE!")
print(f"{'='*50}")
print(f"📄 JSON output: {out_json}")
print(f"🖼️  Images saved: {OUTPUT_DIR}/page_<n>_detections.jpg")
print(f"🎯 Total detections: {len(results)}")
print(f"⏱️  Total VLM time: {total_vlm_time:.2f}s")
print(f"⚡ Average per circle: {total_vlm_time/total_circles:.2f}s" if total_circles > 0 else "")
print(f"{'='*50}")