import subprocess
import sys

try:
    from sklearn.model_selection import train_test_split
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    from sklearn.model_selection import train_test_split

# will finish after cnn_model is done, because that involves feature extraction in cnn_model.py. 
# sequence = run preprocessing -> cnn_model -> train -> etc.