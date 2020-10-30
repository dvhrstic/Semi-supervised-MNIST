import numpy as np
from model import NeuralNetworkModel
from data import Data


def semi_supervised_learning(data):
    X_train_labeled, Y_train_labeled = data.get_labeled_training_data()
    X_train_unlabeled, _ = data.get_unlabeled_training_data()
    X_test, Y_test = data.get_test_data()

    label_learning_model = NeuralNetworkModel()
    label_learning_model.train(X_train_labeled, Y_train_labeled)
    pseudo_labels = label_learning_model.predict(X_train_unlabeled)
    test_scores_labeled_data = label_learning_model.evaluate(X_test, Y_test)

    finalized_prediction_model = NeuralNetworkModel()
    finalized_prediction_model.train(np.concatenate((X_train_unlabeled, X_train_labeled)),
                                     np.concatenate((pseudo_labels, Y_train_labeled)))
    final_labels_test_ = finalized_prediction_model.predict(X_test)
    test_scores_all_data = finalized_prediction_model.evaluate(X_test, Y_test)
    return test_scores_all_data[1]