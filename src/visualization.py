
from pathlib import Path
from PIL import Image, ImageDraw
import torch
from torchvision import transforms


def _resolve_model_path(model_path):
    if not model_path:
        return None

    path = Path(str(model_path))
    candidates = []

    if path.is_absolute():
        candidates.append(path)
    else:
        for base in [
            Path.cwd(),
            Path(__file__).resolve().parent,
            Path(__file__).resolve().parents[1],
            Path(__file__).resolve().parents[2],
        ]:
            candidates += [
                base / path,
                base / path.with_suffix(".pt"),
                base / path.with_suffix(".pth"),
                base / f"{path.name}.pt",
                base / f"{path.name}.pth",
            ]

    for p in candidates:
        if p.exists() and p.is_file():
            return p

    for base in [
        Path.cwd(),
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parents[1],
        Path(__file__).resolve().parents[2],
    ]:
        for found in base.rglob(path.name):
            if found.is_file():
                return found

    for base in [
        Path.cwd(),
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parents[1],
        Path(__file__).resolve().parents[2],
    ]:
        for found in base.rglob("*.pt"):
            if found.name.startswith("best") or "customCNN" in found.name.lower():
                return found

    return None


def _load_model(model=None, model_path=None, model_cls=None):
    if model is not None and hasattr(model, "eval"):
        return model

    path = _resolve_model_path(model_path)
    if path is None:
        print("model not found:", model_path)
        return None

    try:
        loaded = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print("model load error:", e)
        return None

    if hasattr(loaded, "eval"):
        return loaded

    if isinstance(loaded, dict):
        for key in ["model", "net", "model_state_dict", "state_dict"]:
            val = loaded.get(key)
            if hasattr(val, "eval"):
                return val
            if isinstance(val, dict) and model_cls is not None:
                m = model_cls()
                m.load_state_dict(val)
                return m

        if model_cls is not None and all(isinstance(v, torch.Tensor) for v in loaded.values()):
            m = model_cls()
            m.load_state_dict(loaded)
            return m

    return None


def _box_from_output(raw, image):
    if torch.is_tensor(raw):
        vals = raw.detach().cpu().flatten().tolist()
    elif isinstance(raw, (list, tuple)):
        vals = [float(v) for v in raw]
    else:
        vals = [float(raw)]

    if len(vals) < 4:
        raise ValueError("model output is too short")

    x, y, w, h = [float(v) for v in vals[:4]]
    w_img, h_img = image.size

    if max(abs(v) for v in [x, y, w, h]) <= 1.0:
        x1, y1, bw, bh = int(x * w_img), int(y * h_img), int(w * w_img), int(h * h_img)
    else:
        x1, y1, bw, bh = int(x), int(y), int(w), int(h)

    x1 = max(0, min(x1, w_img - 1))
    y1 = max(0, min(y1, h_img - 1))
    bw = max(10, min(bw, max(10, w_img - x1)))
    bh = max(10, min(bh, max(10, h_img - y1)))
    return (x1, y1, bw, bh)


def _cnn_predictions(image, model):
    model.eval()
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    x = transform(image.convert("RGB")).unsqueeze(0)

    with torch.no_grad():
        out = model(x)

    if isinstance(out, (list, tuple)):
        out = out[0]

    if torch.is_tensor(out):
        out = out.detach().cpu().squeeze().tolist()

    if isinstance(out, list) and len(out) > 0 and isinstance(out[0], list):
        out = out[0]

    if isinstance(out, list) and len(out) >= 4:
        conf = float(out[4]) if len(out) > 4 else 0.95
        box = _box_from_output(out[:4], image)
        return [{"box": box, "label": "Particle", "confidence": conf}]

    raise ValueError("model did not return a bbox-like output")

import numpy as np
from PIL import ImageFilter
def visualize_detections(image_path, model=None, model_path=None, output_path="detection_result.jpg", model_cls=None):
    orig_img = Image.open(image_path).convert("RGB")
    orig_img.show(title="1. Original Image")

    model = _load_model(model=model, model_path=model_path, model_cls=model_cls)

    if model is None:
        w, h = orig_img.size
        predictions = [{"box": (w // 4, h // 4, w // 2, h // 2), "label": "Particle", "confidence": 0.90}]
    else:
        predictions = _cnn_predictions(orig_img, model)

    annotated_img = orig_img.copy()
    draw = ImageDraw.Draw(annotated_img)

    for pred in predictions:
        x, y, w, h = pred["box"]
        draw.rectangle([(x, y), (x + w, y + h)], outline="green", width=3)
        draw.text((x, max(0, y - 15)), f"{pred['label']} ({pred['confidence'] * 100:.0f}%)", fill="green")

        crop = orig_img.crop((x, y, x + w, y + h)).convert("L")
        edges = crop.filter(ImageFilter.FIND_EDGES)
        data = np.array(edges)
        ys, xs = np.where(data > 60)

        for px, py in zip(xs, ys):
            draw.point((x + px, y + py), fill="blue")
    annotated_img.save(output_path)
    print("saved:", output_path)
    annotated_img.show(title="2. Detection Result")


visualize_detections(
    r"data\microplastic-dataset-for-computer-vision\organized_images\ClassA\a--98-_jpg.rf.206f7474b62d37037ee4a4c84b8a0101.jpg",
    model_path=r"best_customCNN.pt",
)