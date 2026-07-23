import numpy as np
import io
import PIL.Image as Image
from scipy.ndimage import gaussian_filter


def load_model(path='mlp_weights.npz'):
    data   = np.load(path)
    params = {
        'W1': data['W1'], 'b1': data['b1'],
        'W2': data['W2'], 'b2': data['b2'],
        'W3': data['W3'], 'b3': data['b3'],
    }
    return params

def forward_pass(X, params):
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']
    W3 = params['W3']
    b3 = params['b3']
    Z1 = np.dot(X, W1) + b1
    A1 = np.maximum(0, Z1)  # ReLU activation
    Z2 = np.dot(A1, W2) + b2
    A2 = np.maximum(0, Z2)  # ReLU activation
    Z3 = np.dot(A2, W3) + b3
    exp_scores = np.exp(Z3 - np.max(Z3, axis=1, keepdims=True))  # for numerical stability
    A3 = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # Softmax activation
    return A3 

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert('L')

    # Resize to 280×280 first (in case it comes in different sizes)
    img = img.resize((280, 280), Image.LANCZOS)
    img_array = np.array(img)

    # Invert if white background
    if img_array.mean() > 127:
        img_array = 255 - img_array

    # ── FIND BOUNDING BOX OF DRAWN PIXELS ─────────────────────────
    # Find rows and columns that have any non-zero pixels
    rows = np.any(img_array > 30, axis=1)   # threshold at 30 to ignore noise
    cols = np.any(img_array > 30, axis=0)

    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Add padding around the digit (20% of digit size)
        pad_r = max(10, int((rmax - rmin) * 0.2))
        pad_c = max(10, int((cmax - cmin) * 0.2))

        rmin = max(0, rmin - pad_r)
        rmax = min(img_array.shape[0] - 1, rmax + pad_r)
        cmin = max(0, cmin - pad_c)
        cmax = min(img_array.shape[1] - 1, cmax + pad_c)

        # Crop to bounding box
        img_array = img_array[rmin:rmax+1, cmin:cmax+1]

    # ── RESIZE CROPPED DIGIT TO 28×28 ────────────────────────────
    img_cropped = Image.fromarray(img_array.astype(np.uint8))

    # Make it square first to preserve aspect ratio
    size   = max(img_cropped.size)
    square = Image.new('L', (size, size), 0)   # black background
    offset = ((size - img_cropped.size[0]) // 2,
              (size - img_cropped.size[1]) // 2)
    square.paste(img_cropped, offset)

    #soft blur
    img_array = gaussian_filter(img_array.astype(float), sigma=1)

    # Now resize to 28×28
    final = square.resize((28, 28), Image.LANCZOS)
    img_array = np.array(final).astype(np.float64)

    # Normalize
    img_array = img_array / 255.0

    # Flatten to (1, 784)
    return img_array.reshape(1, 784)

def predict(image_bytes: bytes, params: dict) -> dict:
    # Preprocess
    X = preprocess_image(image_bytes)

    # Forward pass
    probs = forward_pass(X, params)[0]
    # Extract result
    predicted_digit = int(np.argmax(probs))
    confidence      = float(probs[predicted_digit]) * 100
    
    # All class probabilities for display
    all_probs = {
        str(i): round(float(probs[i]) * 100, 2)
        for i in range(10)
    }

    return {
        'digit':      predicted_digit,
        'confidence': round(confidence, 2),
        'all_probs':  all_probs
    }