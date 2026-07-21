from pathlib import Path
import tempfile
import yaml
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]


def _build_absolute_data_yaml(data_yaml_path: Path) -> str:
    data_yaml_path = data_yaml_path.resolve()
    data_config = yaml.safe_load(data_yaml_path.read_text())
    yaml_dir = data_yaml_path.parent

    if isinstance(data_config.get("path"), str):
        data_root = Path(data_config["path"])
        if not data_root.is_absolute():
            data_root = (yaml_dir / data_root).resolve()
        data_config["path"] = str(data_root)
    else:
        data_root = yaml_dir

    for key in ("train", "val", "test"):
        if key in data_config and isinstance(data_config[key], str):
            split_path = Path(data_config[key])
            if not split_path.is_absolute():
                data_config[key] = str((data_root / split_path).resolve())

    temp_file = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w", encoding="utf-8")
    yaml.dump(data_config, temp_file)
    temp_file.close()
    return temp_file.name


def run_quick_test(data_yaml=None, model_path=None, epochs=5, imgsz=640, name="quick_test"):
    data_yaml = Path(data_yaml) if data_yaml else REPO_ROOT / "data.yaml"
    model_path = Path(model_path) if model_path else REPO_ROOT / "yolov8n.pt"

    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found at {data_yaml}")
    if not model_path.exists():
        raise FileNotFoundError(f"model file not found at {model_path}")

    resolved_data_yaml = _build_absolute_data_yaml(data_yaml)
    model = YOLO(str(model_path))
    results = model.train(
        data=resolved_data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        name=name,
    )
    return results


if __name__ == "__main__":
    run_quick_test()

