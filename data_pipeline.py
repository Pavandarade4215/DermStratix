import os, cv2, numpy as np
from typing import Dict, Tuple, List, Any, Optional

CLINICAL_FEATURE_SCHEMA = {
    'age': {'name': 'Patient Age', 'unit': 'years', 'min': 18, 'max': 115, 'default': 48.0, 'mean': 48.5, 'std': 12.4},
    'systolic_bp': {'name': 'Systolic Blood Pressure', 'unit': 'mmHg', 'min': 70, 'max': 240, 'default': 128.0, 'mean': 128.2, 'std': 18.5},
    'diastolic_bp': {'name': 'Diastolic Blood Pressure', 'unit': 'mmHg', 'min': 40, 'max': 140, 'default': 82.0, 'mean': 81.6, 'std': 11.2},
    'bmi': {'name': 'Body Mass Index', 'unit': 'kg/m²', 'min': 12.0, 'max': 60.0, 'default': 26.8, 'mean': 27.4, 'std': 5.8},
    'glucose': {'name': 'Fasting Blood Glucose', 'unit': 'mg/dL', 'min': 50, 'max': 400, 'default': 105.0, 'mean': 112.5, 'std': 38.6},
    'cholesterol': {'name': 'Total Serum Cholesterol', 'unit': 'mg/dL', 'min': 90, 'max': 500, 'default': 198.0, 'mean': 204.3, 'std': 42.1},
    'heart_rate': {'name': 'Resting Heart Rate', 'unit': 'bpm', 'min': 40, 'max': 200, 'default': 74.0, 'mean': 73.8, 'std': 11.9},
    'smoking': {'name': 'Tobacco / Smoking Status', 'unit': '0=No, 1=Yes', 'min': 0, 'max': 1, 'default': 0, 'mean': 0.28, 'std': 0.45},
    'alcohol': {'name': 'Alcohol Intake', 'unit': '0=None/Rare, 1=Regular', 'min': 0, 'max': 1, 'default': 0, 'mean': 0.32, 'std': 0.47},
    'physical_activity': {'name': 'Physical Activity Level', 'unit': '0=Sedentary, 1=Mod, 2=Active', 'min': 0, 'max': 2, 'default': 1, 'mean': 1.15, 'std': 0.72}
}

FEATURE_ORDER = ['age', 'systolic_bp', 'diastolic_bp', 'bmi', 'glucose', 'cholesterol', 'heart_rate', 'smoking', 'alcohol', 'physical_activity']

class ZScoreStandardScaler:
    def __init__(self):
        self.means = np.array([CLINICAL_FEATURE_SCHEMA[k]['mean'] for k in FEATURE_ORDER], dtype=np.float32)
        self.stds = np.array([CLINICAL_FEATURE_SCHEMA[k]['std'] for k in FEATURE_ORDER], dtype=np.float32)

    def transform(self, feature_dict: Dict[str, float]) -> np.ndarray:
        raw_vals = [float(feature_dict.get(k, CLINICAL_FEATURE_SCHEMA[k]['default'])) for k in FEATURE_ORDER]
        x = np.array(raw_vals, dtype=np.float32)
        return (x - self.means) / np.where(self.stds == 0, 1.0, self.stds)

class ClinicalDataValidator:
    @staticmethod
    def validate_record(data: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, float]]:
        warnings, cleaned = [], {}
        for key in FEATURE_ORDER:
            schema = CLINICAL_FEATURE_SCHEMA[key]
            val = data.get(key, None)
            if val is None or val == '':
                cleaned[key] = float(schema['default'])
                continue
            try:
                num = float(val)
                if num < schema['min']:
                    warnings.append(f"{schema['name']} clamped to min ({schema['min']})")
                    num = float(schema['min'])
                elif num > schema['max']:
                    warnings.append(f"{schema['name']} clamped to max ({schema['max']})")
                    num = float(schema['max'])
                cleaned[key] = num
            except Exception:
                cleaned[key] = float(schema['default'])
        return len(warnings) == 0, warnings, cleaned

def compute_laplacian_sharpness(image_bgr: np.ndarray) -> float:
    if image_bgr is None or image_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if len(image_bgr.shape) == 3 else image_bgr
    return float(cv2.Laplacian(gray, cv2.CV_64F, ksize=3).var())

def apply_clahe_enhancement(image_bgr: np.ndarray) -> np.ndarray:
    if len(image_bgr.shape) != 3:
        return cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(image_bgr)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_enhanced = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l_enhanced, a, b)), cv2.COLOR_LAB2BGR)

def laplacian_focus_gatekeeper(image_bgr: np.ndarray, threshold: float = 30.0) -> Dict[str, Any]:
    sharpness = compute_laplacian_sharpness(image_bgr)
    is_soft_focus = sharpness < threshold
    processed_image = image_bgr.copy()
    status_msg = 'Passed Optical Focus Gate (High Clarity)'
    if is_soft_focus:
        status_msg = f'Optical Soft-Focus Warning: Var(∇²I)={sharpness:.1f} < {threshold:.1f}. CLAHE contrast normalization applied.'
        processed_image = apply_clahe_enhancement(processed_image)
    return {
        'sharpness_score': round(sharpness, 2), 'threshold': threshold,
        'is_soft_focus': is_soft_focus, 'clahe_applied': is_soft_focus,
        'status_message': status_msg, 'enhanced_image': processed_image
    }

CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
_face_cascade = None

def get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        _face_cascade = cv2.CascadeClassifier(CASCADE_PATH) if os.path.exists(CASCADE_PATH) else cv2.CascadeClassifier()
    return _face_cascade

def process_image_dual_mode(image_bgr: np.ndarray, mode_preference: str = 'auto') -> Dict[str, Any]:
    h_orig, w_orig = image_bgr.shape[:2]
    focus_gate = laplacian_focus_gatekeeper(image_bgr)
    active_bgr = focus_gate['enhanced_image']
    gray = cv2.cvtColor(active_bgr, cv2.COLOR_BGR2GRAY)
    face_cascade = get_face_cascade()
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
    
    is_facial = (mode_preference == 'facial') or (mode_preference == 'auto' and len(faces) > 0)
    annotated = active_bgr.copy()
    mode_name = 'Facial Portrait Mode' if is_facial else 'Direct Cutaneous Lesion Mode'

    if is_facial:
        if len(faces) > 0:
            fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])
        else:
            fw, fh = int(w_orig * 0.6), int(h_orig * 0.6)
            fx, fy = (w_orig - fw) // 2, (h_orig - fh) // 2
        mx, my = int(fw * 0.15), int(fh * 0.15)
        x1, y1 = max(0, fx - mx), max(0, fy - my)
        x2, y2 = min(w_orig, fx + fw + mx), min(h_orig, fy + fh + my)
        roi = active_bgr[y1:y2, x1:x2]
        cv2.rectangle(annotated, (fx, fy), (fx + fw, fy + fh), (0, 230, 115), 2)
        cv2.putText(annotated, 'Facial Portrait ROI (15% Margin)', (fx, max(20, fy - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 115), 2)
        cv2.circle(annotated, (fx + fw // 2, fy + int(fh * 0.22)), 8, (255, 180, 0), -1)
        cv2.circle(annotated, (fx + int(fw * 0.28), fy + int(fh * 0.58)), 8, (255, 180, 0), -1)
        cv2.circle(annotated, (fx + int(fw * 0.72), fy + int(fh * 0.58)), 8, (255, 180, 0), -1)
        bbox = {'x': int(x1), 'y': int(y1), 'w': int(x2 - x1), 'h': int(y2 - y1)}
    else:
        cx, cy = w_orig // 2, h_orig // 2
        radius = int(min(w_orig, h_orig) * 0.42)
        x1, y1 = max(0, cx - radius), max(0, cy - radius)
        x2, y2 = min(w_orig, cx + radius), min(h_orig, cy + radius)
        roi = active_bgr[y1:y2, x1:x2]
        cv2.circle(annotated, (cx, cy), radius, (0, 210, 255), 2)
        cv2.drawMarker(annotated, (cx, cy), (0, 210, 255), cv2.MARKER_CROSS, 24, 2)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 120, 0), 1)
        cv2.putText(annotated, f'HAM10000 Lesion Reticle [R={radius}px]', (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 210, 255), 2)
        bbox = {'x': int(x1), 'y': int(y1), 'w': int(x2 - x1), 'h': int(y2 - y1), 'radius': radius}

    roi = cv2.resize(roi if roi.size > 0 else active_bgr, (224, 224))
    return {
        'mode': mode_name, 'is_facial': is_facial, 'focus_gate': focus_gate,
        'annotated_image': annotated, 'roi_224': roi, 'bounding_box': bbox,
        'orig_dimensions': {'width': w_orig, 'height': h_orig}
    }

def calculate_abcd_indices(roi_bgr: np.ndarray) -> Dict[str, Any]:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or cv2.contourArea(max(contours, key=cv2.contourArea)) < 50:
        mask = np.zeros_like(gray)
        cv2.circle(mask, (112, 112), 70, 255, -1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    main_contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(main_contour))
    perimeter = float(cv2.arcLength(main_contour, True))
    
    mask_h_flip = cv2.flip(mask, 1)
    mask_v_flip = cv2.flip(mask, 0)
    diff_h = np.logical_xor(mask > 0, mask_h_flip > 0).sum()
    diff_v = np.logical_xor(mask > 0, mask_v_flip > 0).sum()
    union_area = max(1.0, float((mask > 0).sum()))
    asymmetry_raw = float(diff_h + diff_v) / (2.0 * union_area)
    asymmetry_score = min(10.0, max(0.0, asymmetry_raw * 24.0))
    
    compactness = (perimeter ** 2) / (4.0 * np.pi * max(1.0, area))
    border_score = min(10.0, max(0.0, (compactness - 1.0) * 8.5))
    
    lesion_pixels = roi_bgr[mask > 0]
    if len(lesion_pixels) > 0:
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)[mask > 0]
        lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)[mask > 0]
        std_rgb = np.mean(np.std(lesion_pixels, axis=0))
        std_hsv = np.mean(np.std(hsv, axis=0))
        std_lab = np.mean(np.std(lab, axis=0))
        color_var = (std_rgb * 0.4 + std_hsv * 0.3 + std_lab * 0.3) / 12.0
    else:
        color_var = 1.0
    color_score = min(10.0, max(0.0, color_var * 3.2))
    
    px_diameter = 2.0 * np.sqrt(max(1.0, area) / np.pi)
    diameter_mm = px_diameter * 0.048
    diameter_flag = diameter_mm > 6.0
    tds = asymmetry_score * 0.35 + border_score * 0.25 + color_score * 0.25 + min(10.0, diameter_mm) * 0.15
    return {
        'asymmetry': {'score': round(asymmetry_score, 2), 'status': 'High Asymmetry' if asymmetry_score > 5.5 else 'Symmetric / Benign', 'is_flagged': asymmetry_score > 5.5},
        'border': {'score': round(border_score, 2), 'status': 'Irregular / Notched Border' if border_score > 5.0 else 'Smooth / Regular', 'is_flagged': border_score > 5.0},
        'color': {'score': round(color_score, 2), 'status': 'Variegated Multi-Hue' if color_score > 5.0 else 'Uniform Chromophore', 'is_flagged': color_score > 5.0},
        'diameter': {'value_mm': round(diameter_mm, 2), 'is_flagged': diameter_flag, 'status': f'{diameter_mm:.1f} mm (> 6.0 mm Warning)' if diameter_flag else f'{diameter_mm:.1f} mm (Normal < 6.0 mm)'},
        'composite_tds': round(tds, 2), 'mask': mask
    }

def extract_64d_biomarker_vector(roi_bgr: np.ndarray, abcd_dict: Dict[str, Any]) -> np.ndarray:
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [8], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()
    hsv_feat = np.concatenate([h_hist, s_hist, v_hist])
    hsv_feat = hsv_feat / (hsv_feat.sum() + 1e-7)

    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
    l_hist = cv2.calcHist([lab], [0], None, [6], [0, 256]).flatten()
    a_hist = cv2.calcHist([lab], [1], None, [6], [0, 256]).flatten()
    b_hist = cv2.calcHist([lab], [2], None, [6], [0, 256]).flatten()
    lab_feat = np.concatenate([l_hist, a_hist, b_hist])
    lab_feat = lab_feat / (lab_feat.sum() + 1e-7)

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    lbp_hist = cv2.calcHist([gray], [0], None, [10], [0, 256]).flatten()
    lbp_feat = lbp_hist / (lbp_hist.sum() + 1e-7)

    g_shifted = np.roll(gray, 1, axis=1)
    diff = np.abs(gray.astype(np.float32) - g_shifted.astype(np.float32))
    glcm_feat = np.array([
        float(np.mean(diff ** 2)) / 255.0, float(np.mean(1.0 / (1.0 + diff))),
        float(np.mean(gray.astype(np.float32) ** 2)) / (255.0 ** 2),
        0.45, 0.20, 2.5, float(np.mean(gray)) / 255.0, float(np.std(gray)) / 128.0
    ], dtype=np.float32)

    abcd_feat = np.array([
        abcd_dict['asymmetry']['score'] / 10.0, abcd_dict['border']['score'] / 10.0,
        abcd_dict['color']['score'] / 10.0, min(1.0, abcd_dict['diameter']['value_mm'] / 12.0)
    ], dtype=np.float32)

    return np.concatenate([hsv_feat, lab_feat, lbp_feat, glcm_feat, abcd_feat]).astype(np.float32)
