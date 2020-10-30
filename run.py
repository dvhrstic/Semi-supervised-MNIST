
import numpy as np
from model import NeuralNetworkModel
from ssl import semi_supervised_learning
from data import Data


data = Data()
test_accuracy_score = semi_supervised_learning(data)
print("Test data accuracy for model trained on all data ",
      test_accuracy_score)
