import csv
import torch

#saves the models weights when current_loss is less than best_loss, and also saves the best_loss value
#im lowk cooking rn
class ModelCheckpoint:
  def __init__(self, filepath='customCNN.pt'):
    self.filepath = filepath
    self.best_loss = float('inf')  

  def check_and_save(self, current_loss, model):
    if current_loss < self.best_loss:
      self.best_loss = current_loss
      torch.save(model.state_dict(), self.filepath)
      print(f' Saved new best model to {self.filepath}')

#stops training if the current_loss does not improve for a specified number of epochs
#note from anishka: for the self.should_stop thing, in train.py make sure when using this class you manually check the value of self.should_stop after each epoch and break the training loop if it's True, cause this class doesn't have direct control over the training loop.
#dam that was a long note and ii actually had perfect grammar guys omg
class EarlyStopping:
  def __init__(self, epochs=3):
    self.epochs = epochs
    self.best_loss = float('inf')
    self.counter = 0
    self.should_stop = False

  def check_stop(self, current_loss):
    if current_loss < self.best_loss:
      self.best_loss = current_loss
      self.counter = 0  # Reset counter on improvement
    else:
      self.counter += 1
      print(f' No improvement. Early stopping counter: {self.counter}/{self.epochs}')
      if self.counter >= self.epochs:
        self.should_stop = True

#logs training metrics to CSV file for analysing and stuff ;>
def log_results(epoch, train_loss, val_loss, lr):
  filepath = 'src\\logs\\training_logs.csv'
  with open(filepath, 'a', newline='') as f:
    csv.writer(f).writerow([epoch, train_loss, val_loss, lr])