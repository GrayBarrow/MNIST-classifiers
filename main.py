from data_loader import load_mnist
from knn_classifier import train_and_evaluate_knn
from naive_bayes_classifier import train_and_evaluate_naive_bayes
from linear_nn_classifier import train_and_evaluate_linear_nn
from mlp_classifier import train_and_evaluate_MLP
from cnn_classifier import train_and_evaluate_cnn

def main():
    X_train, X_test, y_train, y_test = load_mnist('mnist_dataset.npz')
    
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
