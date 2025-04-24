import streamlit as st
import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from skimage import filters, segmentation, color, exposure
from sklearn.cluster import KMeans
import io

st.set_page_config(page_title="CosmoVision Dashboard", layout="wide")
st.title("🪐 CosmoVision: Space Image Processing Lab")

# --- Helper functions ---
def load_image(uploaded_file):
    image = Image.open(uploaded_file)
    return np.array(image)

def display_image(title, img, cmap=None):
    st.subheader(title)
    st.image(img, use_column_width=True, clamp=True, channels="RGB" if img.ndim == 3 else "GRAY")

def apply_filter(img, mode):
    if mode == "Mean":
        return cv2.blur(img, (5, 5))
    elif mode == "Max":
        return cv2.dilate(img, np.ones((5, 5), np.uint8))
    elif mode == "Min":
        return cv2.erode(img, np.ones((5, 5), np.uint8))

def edge_detection(img, method):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if method == "Prewitt":
        kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        x = cv2.filter2D(gray, -1, kernelx)
        y = cv2.filter2D(gray, -1, kernely)
        return cv2.convertScaleAbs(x + y)
    elif method == "Roberts":
        kernelx = np.array([[1, 0], [0, -1]])
        kernely = np.array([[0, 1], [-1, 0]])
        x = cv2.filter2D(gray, -1, kernelx)
        y = cv2.filter2D(gray, -1, kernely)
        return cv2.convertScaleAbs(x + y)
    elif method == "LoG":
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        return cv2.Laplacian(blur, cv2.CV_64F)

def frequency_filter(img, type_):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    dft = np.fft.fft2(gray)
    dft_shift = np.fft.fftshift(dft)
    rows, cols = gray.shape
    crow, ccol = rows//2, cols//2
    mask = np.zeros((rows, cols), np.float32)
    D0 = 50  # Cutoff

    for i in range(rows):
        for j in range(cols):
            d = np.sqrt((i - crow)**2 + (j - ccol)**2)
            if type_ in ["IHPF", "ILPF"]:
                mask[i, j] = 0 if d < D0 else 1 if type_ == "IHPF" else 1
            elif type_ in ["BHPF", "BLPF"]:
                n = 2
                mask[i, j] = 1 / (1 + (D0 / (d + 1e-5))**(2*n)) if "BLPF" in type_ else 1 - 1 / (1 + (d / D0)**(2*n))
            elif type_ in ["GHPF", "GLPF"]:
                mask[i, j] = 1 - np.exp(-(d**2)/(2*(D0**2))) if "GHPF" in type_ else np.exp(-(d**2)/(2*(D0**2)))

    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.abs(np.fft.ifft2(f_ishift))
    return img_back.astype(np.uint8)

def jpeg_compress(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_dct = cv2.dct(np.float32(gray)/255.0)
    img_idct = cv2.idct(img_dct) * 255.0
    return np.clip(img_idct, 0, 255).astype(np.uint8)

def manipulate_channel(img, color_space, channel):
    if color_space == "RGB":
        img[:, :, channel] = 0
        return img
    elif color_space == "YCbCr":
        ycbcr = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        ycbcr[:, :, channel] = 0
        return cv2.cvtColor(ycbcr, cv2.COLOR_YCrCb2RGB)

def convert_color_space(img, to):
    if to == "HSV":
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif to == "HSI":
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        return hsv  # approximate
    elif to == "CMY":
        cmy = 255 - img
        return cmy

def saturation_control(img, level):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * level, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def intensity_slicing(img, min_val, max_val):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sliced = np.where((gray >= min_val) & (gray <= max_val), 255, 0)
    return sliced.astype(np.uint8)

def segment_image(img, method):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if method == "Thresholding":
        _, seg = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
        return seg
    elif method == "K-means":
        Z = img.reshape((-1, 3))
        Z = np.float32(Z)
        _, labels, centers = cv2.kmeans(Z, 3, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        return centers[labels.flatten()].reshape(img.shape)

# --- Sidebar: Image Upload ---
st.sidebar.title("📷 Upload Image")
uploaded_file = st.sidebar.file_uploader("Upload Earth/Space image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = load_image(uploaded_file)
    display_image("Original Image", img)

    # --- Operation selection ---
    operation = st.sidebar.selectbox("Choose Operation", [
        "Basic Filters", "Edge Detection", "Frequency Filters", "Color Manipulation",
        "Image Compression", "Segmentation", "Saturation", "Intensity Slicing"
    ])

    # --- Basic Filters ---
    if operation == "Basic Filters":
        filter_type = st.sidebar.selectbox("Filter Type", ["Mean", "Max", "Min"])
        result = apply_filter(img, filter_type)
        display_image(f"{filter_type} Filter", result)

    # --- Edge Detection ---
    elif operation == "Edge Detection":
        method = st.sidebar.selectbox("Method", ["Prewitt", "Roberts", "LoG"])
        result = edge_detection(img, method)
        display_image(f"{method} Edge Detection", result)

    # --- Frequency Filters ---
    elif operation == "Frequency Filters":
        ftype = st.sidebar.selectbox("Filter Type", ["IHPF", "ILPF", "BHPF", "BLPF", "GHPF", "GLPF"])
        result = frequency_filter(img, ftype)
        display_image(f"{ftype} Result", result)

    # --- Image Compression ---
    elif operation == "Image Compression":
        result = jpeg_compress(img)
        display_image("JPEG-like Decompressed Image", result)

    # --- Color Manipulation ---
    elif operation == "Color Manipulation":
        task = st.sidebar.selectbox("Task", ["Remove Channel", "Convert Color Model"])
        if task == "Remove Channel":
            space = st.sidebar.selectbox("Color Space", ["RGB", "YCbCr"])
            channel = st.sidebar.selectbox("Channel", [0, 1, 2])
            result = manipulate_channel(img.copy(), space, channel)
            display_image(f"{space} with Channel {channel} removed", result)
        else:
            to_space = st.sidebar.selectbox("Convert To", ["HSV", "HSI", "CMY"])
            result = convert_color_space(img.copy(), to_space)
            display_image(f"Converted to {to_space}", result)

    # --- Saturation Control ---
    elif operation == "Saturation":
        level = st.sidebar.slider("Saturation Level", 0.0, 2.0, 1.0)
        result = saturation_control(img.copy(), level)
        display_image("Saturation Adjusted", result)

    # --- Intensity Slicing ---
    elif operation == "Intensity Slicing":
        minv = st.sidebar.slider("Min Value", 0, 255, 100)
        maxv = st.sidebar.slider("Max Value", 0, 255, 200)
        result = intensity_slicing(img.copy(), minv, maxv)
        display_image("Intensity Sliced Image", result)

    # --- Segmentation ---
    elif operation == "Segmentation":
        seg_method = st.sidebar.selectbox("Segmentation Type", ["Thresholding", "K-means"])
        result = segment_image(img.copy(), seg_method)
        display_image(f"{seg_method} Segmentation", result)