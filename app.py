import streamlit as st
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image
from facenet_pytorch import MTCNN
import torch
import io

# -------- MTCNN FACE DETECTION --------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=True, device=device)

def detect_faces(image):
    boxes, _ = mtcnn.detect(image)
    if boxes is None:
        return []
    return [tuple(map(int, box)) for box in boxes]

# -------- LIGHT LEAK FUNCTIONS --------
def random_color():
    palette = [
        np.array([1.0, np.random.uniform(0.2, 0.4), 0.0]),
        np.array([1.0, 0.0, np.random.uniform(0.2, 0.6)]),
        np.array([np.random.uniform(0.8, 1.0), np.random.uniform(0.3, 0.6), 0.0]),
        np.array([np.random.uniform(0.7, 1.0), 0.0, np.random.uniform(0.3, 0.6)]),
    ]
    return palette[np.random.randint(len(palette))], palette[np.random.randint(len(palette))]

def generate_light_leak_mask(shape, faces=None, num_leaks=3):
    h, w = shape[:2]
    final_mask = np.zeros((h, w), dtype=np.float32)

    y, x = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2
    distance = np.sqrt((x - cx)**2 + (y - cy)**2)
    prob_map = (distance / distance.max())**2

    face_mask = np.ones((h, w), dtype=np.float32)
    if faces:
        for (x1, y1, x2, y2) in faces:
            face_mask[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = 0.0
        prob_map *= face_mask

    prob_map /= prob_map.sum() + 1e-8
    coords = np.argwhere(prob_map > 0)

    for _ in range(num_leaks):
        for _ in range(30):
            idx = np.random.choice(len(coords))
            cy, cx = coords[idx]
            temp = np.zeros_like(final_mask)
            shape_type = np.random.choice(['ellipse', 'streak', 'radial'])

            if shape_type == 'ellipse':
                axes = (np.random.randint(w//10, w//3), np.random.randint(h//10, h//3))
                angle = np.random.uniform(0, 360)
                cv2.ellipse(temp, (cx, cy), axes, angle, 0, 360, color=1, thickness=-1)

            elif shape_type == 'streak':
                orientation = np.random.choice(['horizontal', 'vertical'])
                temp_streak = np.zeros_like(temp)

                length = np.random.randint(w // 4, w)
                thickness = np.random.randint(20, 80)

                if orientation == 'horizontal':
                    x1 = max(0, cx - length // 2)
                    x2 = min(w - 1, cx + length // 2)
                    y1 = y2 = np.clip(cy, 0, h - 1)
                    cv2.line(temp_streak, (x1, y1), (x2, y2), color=1, thickness=thickness)
                else:  # vertical
                    y1 = max(0, cy - length // 2)
                    y2 = min(h - 1, cy + length // 2)
                    x1 = x2 = np.clip(cx, 0, w - 1)
                    cv2.line(temp_streak, (x1, y1), (x2, y2), color=1, thickness=thickness)

                sigma = np.random.uniform(15, 40)
                temp = gaussian_filter(temp_streak.astype(np.float32), sigma=sigma)

            elif shape_type == 'radial':
                radial = np.zeros_like(temp)
                cv2.circle(radial, (cx, cy), np.random.randint(h//8, h//2), color=1, thickness=-1)
                temp = gaussian_filter(radial, sigma=np.random.uniform(60, 160))

            # Only add if the shape is fully outside faces
            if np.all(temp * face_mask == temp):
                final_mask += temp * np.random.uniform(0.5, 1.2)
                break

    blurred = gaussian_filter(final_mask, sigma=np.random.uniform(60, 180))
    return np.clip(blurred / (blurred.max() + 1e-8) * np.random.uniform(0.4, 0.9), 0, 1)

def generate_colored_leak(mask):
    h, w = mask.shape
    mask = mask[..., None]
    color_a, color_b = random_color()
    t = np.random.rand()
    gradient_color = color_a + t * (color_b - color_a)
    gradient_color = np.clip(gradient_color * np.random.uniform(1.0, 1.6), 0, 1)
    noise = np.random.normal(scale=np.random.uniform(0.01, 0.04), size=mask.shape)
    return np.clip(mask * gradient_color + noise, 0, 1)

def blend_screen(base, leak):
    return 1 - (1 - base) * (1 - leak)

def apply_light_leak_pil(pil_img):
    rgb_img = np.array(pil_img.convert('RGB'))
    img_float = rgb_img / 255.0
    faces = detect_faces(rgb_img)

    combined_leak = np.zeros_like(img_float)
    for _ in range(np.random.randint(2, 5)):
        mask = generate_light_leak_mask(rgb_img.shape, faces=faces, num_leaks=1)
        colored_leak = generate_colored_leak(mask)
        combined_leak = blend_screen(combined_leak, colored_leak)

    result = blend_screen(img_float, combined_leak)
    result = np.clip(result, 0, 1)
    result_img = (result * 255).astype(np.uint8)
    return Image.fromarray(result_img), len(faces)

# -------- STREAMLIT UI --------
st.set_page_config(page_title="Light Leak Generator", layout="wide")

st.title("🎞️ Light Leak Generator (Multiple Images)")
st.markdown("Upload multiple images to apply artistic light leak effects. Faces will be preserved.")

uploaded_files = st.file_uploader("Upload Image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    if st.button("Apply Light Leak Effect"):
        for uploaded_file in uploaded_files:
            st.markdown(f"#### {uploaded_file.name}")
            col1, col2 = st.columns(2)

            with col1:
                input_img = Image.open(uploaded_file)
                st.image(input_img, caption="Original", use_container_width=True)

            with col2:
                with st.spinner("Processing..."):
                    output_img, face_count = apply_light_leak_pil(input_img)
                    st.image(output_img, caption=f"With Light Leak (Faces Detected: {face_count})", use_container_width=True)

                    img_byte_arr = io.BytesIO()
                    output_img.save(img_byte_arr, format='JPEG')
                    st.download_button(
                        "Download",
                        data=img_byte_arr.getvalue(),
                        file_name=f"leak_{uploaded_file.name}",
                        mime="image/jpeg",
                        key=uploaded_file.name,
                    )