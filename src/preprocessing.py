import shutil  #  shutil so images can be copied into the organized class folders.
from pathlib import Path  # path so the script uses portable filesystem paths instead of absolute local paths.
from typing import Optional, Tuple  #  the typing helpers used by the function annotations.

import pandas as pd  
from PIL import Image  # PIL so the image width and height can be read directly from each file.


REPO_ROOT = Path(__file__).resolve().parent.parent  #resolves the repository root from the script location so the paths work on GitHub and any machine.
DEFAULT_DATASET_ROOT = REPO_ROOT / "new_data" / "archive"  #uses the new dataset root inside the repository.
DEFAULT_OUTPUT_DIR = DEFAULT_DATASET_ROOT / "organized_images"  #sets the default output folder for the organized image classes.


#defines the helper that maps a filename prefix to a class folder.
def class_name_from_filename(filename: str) -> Optional[str]:  # Accepts a filename and returns the matching class name if one exists.
    name = Path(filename).name  # extracts only the file name so prefixes are checked consistently.
    if name.startswith("a--"):  # matches the ClassA prefix used by the dataset naming convention.
        return "ClassA"  # returns the ClassA folder name.
    if name.startswith("b--"):  #etc
        return "ClassB"  
    if name.startswith("c--"):  
        return "ClassC"  
    if name.startswith("d--"):  
        return "ClassD"  
    if name.startswith("f--"):  
        return "ClassF"
    return None  # returns None when the filename does not match one of the known class prefixes.


# defines the main function that prepares a class-based dataset from the annotation CSV.
def prepare_classified_dataset(dataset_root: Path | str = DEFAULT_DATASET_ROOT, output_dir: Path | str = DEFAULT_OUTPUT_DIR, annotations_path: Path | str | None = None, include_unlabeled: bool = True) -> Tuple[Path, Path]:  # Accepts the dataset root, output folder, annotation file, and whether unlabeled images should be included.
    dataset_root = Path(dataset_root)  #dataset root -> pathlib Path object.
    output_dir = Path(output_dir)  # output directory -> pathlib Path object.
    output_dir.mkdir(parents=True, exist_ok=True)  # creates the output directory if it does not exist yet.
    manifest_path = output_dir / "dataset_manifest.csv"  #creates the manifest path inside the output directory.

    if annotations_path is not None:  # keeps the parameter available for compatibility but does not use it as the main source of truth anymore.
        annotations_path = Path(annotations_path)  # converts the passed path into a pathlib Path object.

    image_root = dataset_root / "train"  # points to the new training image directory inside the dataset root.
    if not image_root.exists():  # raises an error if the original images folder is missing.
        raise FileNotFoundError(f"Image root not found: {image_root}")  

    class_dirs = {  # defines the destination folders for each class and the background folder.
        "ClassA": output_dir / "ClassA",  # maps the ClassA label to the ClassA output folder.
        "ClassB": output_dir / "ClassB",  # etc etc etc
        "ClassC": output_dir / "ClassC",  
        "ClassD": output_dir / "ClassD",  
        "ClassF": output_dir / "ClassF",  
        "Background": output_dir / "Background",  #maps unlabeled images to the Background output folder.
    }
    for class_dir in class_dirs.values():  # creates every output subfolder so the dataset structure exists.
        class_dir.mkdir(parents=True, exist_ok=True)  # makes class directory if it is missing.

    manifest_rows = []  # empty list that will store the manifest records.
    copied_count = 0  #tracks how many images were copied into the organized folders.

    all_image_files = [p for p in image_root.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]  #finds every image file under the dataset images folder.

    for source_path in all_image_files:  # iterates through every discovered image file.
        filename = source_path.name  # Stores the file name.
        class_name = class_name_from_filename(filename)  # maps the filename prefix to a class.
        if class_name is None:  # if the image does not match one of the known class prefixes, it is treated as background when enabled.
            if include_unlabeled:  # sends unlabeled images to the Background folder when enabled.
                destination_dir = class_dirs["Background"]  #selects the Background destination folder.
            else:  # skips unlabeled images when the option is disabled.
                continue  
        else:  # uses the detected class folder when the filename matches a known prefix.
            destination_dir = class_dirs[class_name]  #Selects the class destination folder.

        destination_path = destination_dir / filename  #builds the destination path for the image file.
        if not destination_path.exists():  # copies the file ONLY if it is not already present in the destination folder.
            shutil.copy2(source_path, destination_path)  #copies the image while preserving metadata.

        with Image.open(source_path) as image:  # opens the source image so its dimensions can be written into the manifest.
            width, height = image.size  # reads the actual image width and height from the file.

        manifest_rows.append({  # records the copy operation in the manifest row list.
            "filename": filename,  #stores image file name.
            "width": width,  #stores the image width in pixels.
            "height": height,  #stores the image height in pixels.
            "class": destination_dir.name,  #stores the destination class name.
            "xmin": 0,  #stores the left edge of the box.
            "ymin": 0,  #stores the upper edge of the box.
            "xmax": width,  #stores the right edge of the box.
            "ymax": height,  #stores the lower edge of the box.
        })
        copied_count += 1  # increments the image counter after a successful copy.

    manifest_df = pd.DataFrame(manifest_rows)  # builds a DataFrame from the manifest rows.
    manifest_df = manifest_df[["filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"]]  # keeps the manifest columns aligned with the required object-detection layout.
    manifest_df.to_csv(manifest_path, index=False)  # writes the manifest to a CSV file for easy inspection.

    print(f"Copied {copied_count} images into organized class folders.")  # Prints the total number of copied images.
    print(f"Manifest written to {manifest_path}")  # prints the manifest path for the user.
    return output_dir, manifest_path  #returns the output directory and manifest path.


# runs the preparation function when the script is executed directly.
if __name__ == "__main__":  #checks whether the file is running as a script rather than being imported.
    prepare_classified_dataset()  #runs the dataset preparation step with the default settings.
