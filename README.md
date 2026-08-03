# TEAM NAME: Triple-T

Collaborators: Anya Saravanan, Prerita Singh, and Anishka Balla

**Project Title: Real-Time Microplastic Detection**



We developed our own custom CNN using micro-particle microscopy images from a pre-existing Kaggle dataset to isolate plastic pollutants in drinking water. The network identifies the exact contours of microscopic synthetic fibers. A possible application of this allows for automated water filtration systems to trap dangerous particles before consumption.

The dataset we used -> ((https://www.kaggle.com/datasets/imtkaggleteam/microplastic-dataset-for-computer-vision/data)): This dataset contains labeled microscopy images specifically isolated from urban water sources (including tap water, filtered water, and greywater) using membrane filtration and staining. It is tailored for automated computer vision detection and morphological classification.

Limitations: The current model is trained on a relatively small public dataset, which limits generalization. Future work includes collecting a larger and more diverse dataset, experimenting with architectures such as YOLO or Faster R-CNN, and improving localization accuracy through additional hyperparameter tuning.


