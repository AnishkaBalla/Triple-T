import tempfile  # imports tempfile so a temporary dataset can be created for testing.
import unittest  #imports unittest so the helper functions can be verified with automated tests.
from pathlib import Path  

from PIL import Image  # imports Pillow so the temporary test image can be written as a real image file.

from src.preprocessing import class_name_from_filename, prepare_classified_dataset  # Imports the preprocessing helpers that need to be tested.


#defines a test case for the preprocessing helpers.
class PreprocessingTests(unittest.TestCase):  # creates a unittest class for the preprocessing logic.
    def test_class_name_from_filename(self):  #verifies that a filename prefix maps to the expected class folder.
        self.assertEqual(class_name_from_filename("a--1-_jpg.rf.test.jpg"), "ClassA")  #confirms that the ClassA prefix resolves to ClassA.
        self.assertEqual(class_name_from_filename("c--34-_jpg.rf.test.jpg"), "ClassC")  # confirms that the ClassC prefix resolves to ClassC.
        self.assertIsNone(class_name_from_filename("100_jpg.rf.test.jpg"))  #Confirms that a non-prefixed image does not get a class name.

    def test_prepare_classified_dataset_copies_labeled_images(self):  # verifies that labeled images are copied into the organized class folders.
        with tempfile.TemporaryDirectory() as tmpdir:  # temporary directory for the synthetic dataset.
            root = Path(tmpdir)  #stores the temporary root path.
            images_dir = root / "images"  # creates the images folder inside the temporary dataset.
            output_dir = root / "organized_images"  # creates the output folder inside the temporary dataset.
            images_dir.mkdir(parents=True)  #images directory.
            output_dir.mkdir(parents=True)  #output directory.

            source_file = images_dir / "a--1-_jpg.rf.test.jpg"  #creates a fake source image file in the temporary images directory.
            Image.new("RGB", (32, 32), color="white").save(source_file)  #writes a valid image file so the preprocessing step can read its size.

            manifest_path = output_dir / "dataset_manifest.csv"  #creates the manifest CSV path inside the temporary output directory.

            result_dir, manifest_path = prepare_classified_dataset(  #  preprocessing function on the temporary dataset.
                dataset_root=root,  #Passes temporary dataset root.
                output_dir=output_dir,  #Passes temporary output directory.
                annotations_path=manifest_path,  #passes the manifest path to the preprocessing function.
            )

            self.assertTrue((result_dir / "ClassA" / source_file.name).exists())  #confirms that the image was copied into the ClassA folder.
            self.assertTrue(manifest_path.exists())  #confims that the manifest file was created.
            manifest = manifest_path.read_text(encoding="utf-8")  #reads the manifest file contents.
            self.assertIn("filename", manifest)  #Confirms that the manifest contains the filename column.
            self.assertIn("ClassA", manifest)  #Confirms that the manifest includes the expected class label.


# runs the unittest suite when the file is executed directly.
if __name__ == "__main__":  #Checks whether the test file is being run as a script.
    unittest.main()  #Executes unit tests.
