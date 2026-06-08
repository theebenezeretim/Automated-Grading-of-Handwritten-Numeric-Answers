import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# 1. Page Config & Header
st.set_page_config(page_title="AI Handwritten Grading System", layout="wide")
st.title("📝 AI Handwritten Digit Grading Portal")
st.write("Upload a student's answer sheet image to dynamically extract, recognize, and grade handwritten answers.")

# 2. Hardcoded Answer Box Coordinates & Answer Key (From Your Logic)
BOXES = {
    "A1": (50, 50, 150, 100),
    "A2": (50, 150, 150, 200),
    "A3": (50, 250, 150, 300),
    "A4": (50, 350, 150, 400),
    "A5": (50, 450, 150, 500),
    "A6": (50, 550, 150, 600),
    "A7": (50, 650, 150, 700),
}

ANSWER_KEY = {
    "A1": 45, "A2": 11, "A3": 4, "A4": 2, "A5": 3, "A6": 25, "A7": 16
}

# 3. Cache the Model Initialization
@st.cache_resource
def load_grading_model():
    try:
        return load_model("handwritten_grading_model.keras")
    except Exception as e:
        st.error("Could not find 'handwritten_grading_model.keras' in your repository root directory. Please make sure the trained model file is uploaded.")
        return None

model = load_grading_model()

# 4. Main Sidebar Configuration
st.sidebar.header("📋 Current Answer Key")
for question, correct_val in ANSWER_KEY.items():
    st.sidebar.text(f"{question}: {correct_val}")

# 5. File Upload Handler
uploaded_file = st.file_uploader("Choose a student answer sheet image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    # Convert uploaded file into OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Layout splits: Original Sheet view vs Processing & Scoring
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📸 Uploaded Answer Sheet")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
        
    with col2:
        st.subheader("🔍 Processing & Grading Results")
        
        score = 0
        predictions = {}
        
        # Process each answer box bounding coordinate
        for key, (x1, y1, x2, y2) in BOXES.items():
            # Check boundaries safely
            h_img, w_img, _ = img.shape
            if y2 <= h_img and x2 <= w_img:
                crop = img[y1:y2, x1:x2]
                
                # Image adjustments (Grayscale + Threshold)
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
                
                # Extract handwritten contours
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
                
                predicted_digits = []
                
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    # Filter tiny noise fragments
                    if w > 5 and h > 5:
                        digit_image = thresh[y:y+h, x:x+w]
                        
                        # Pad and resize to 28x28 (MNIST standard layout)
                        digit_resized = cv2.resize(digit_image, (28, 28))
                        digit_normalized = digit_resized / 255.0
                        digit_reshaped = digit_normalized.reshape(1, 28, 28, 1)
                        
                        # Inference Execution
                        prediction = model.predict(digit_reshaped, verbose=0)
                        number = prediction.argmax()
                        predicted_digits.append(str(number))
                
                # Reconstruct multiple digits (e.g. ['4', '5'] -> 45)
                if predicted_digits:
                    final_answer = int("".join(predicted_digits))
                else:
                    final_answer = 0  # Blank box adjustment
                
                predictions[key] = final_answer
                
                # Check performance against baseline answer sheets
                if predictions[key] == ANSWER_KEY[key]:
                    score += 1
                    st.write(f"✅ **{key}**: Predicted `{final_answer}` | Correct!")
                else:
                    st.write(f"❌ **{key}**: Predicted `{final_answer}` | Expected `{ANSWER_KEY[key]}`")
            else:
                st.warning(f"⚠️ Box {key} falls outside image bounds.")

        # Metric Displays
        st.markdown("---")
        final_percentage = (score / len(ANSWER_KEY)) * 100
        st.metric(label="🏆 Total Score", value=f"{score} / {len(ANSWER_KEY)}", delta=f"{final_percentage:.1f}%")
