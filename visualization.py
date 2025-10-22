import matplotlib.pyplot as plt
from sklearn import metrics

def plot_confusion_matrix(y_test, y_pred):
    cm = metrics.confusion_matrix(y_test, y_pred)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
    cm_display.plot()
    plt.show()

def plot_training_curves(total_epoch_loss_train, total_epoch_loss_test, total_epoch_acc_train, total_epoch_acc_test):
    #loss plot
    plt.figure()
    plt.plot(total_epoch_loss_train, label="Train Loss")
    plt.plot(total_epoch_loss_test, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    #accuracy plot
    plt.figure()
    plt.plot(total_epoch_acc_train, label="Train Accuracy")
    plt.plot(total_epoch_acc_test, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
