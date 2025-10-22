from data_loader import load_mnist
from Classifiers.knn_classifier import train_and_evaluate_knn
from Classifiers.naive_bayes_classifier import train_and_evaluate_naive_bayes
from Classifiers.linear_nn_classifier import train_and_evaluate_linear_nn
from Classifiers.mlp_classifier import train_and_evaluate_MLP
from Classifiers.cnn_classifier import train_and_evaluate_cnn

def main():
    X_train, X_test, y_train, y_test = load_mnist('data/mnist_dataset.npz')
    
    print("Training and Evaluating KNN Classifier")
    train_and_evaluate_knn(X_train, X_test, y_train, y_test)

    print("Training and Evaluating Naive Bayes Classifier")
    train_and_evaluate_naive_bayes(X_train, X_test, y_train, y_test)
    
    print("Training and Evaluating Linear NN Classifier")
    train_and_evaluate_linear_nn(X_train, X_test, y_train, y_test)
    
    print("Training and Evaluating Non-Linear NN Classifier")
    train_and_evaluate_MLP(X_train, X_test, y_train, y_test)
    
    print("Training and Evaluating CNN Classifier")
    train_and_evaluate_cnn(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
