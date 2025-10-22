import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from preprocessing import preprocess_for_nn
from visualization import plot_training_curves, plot_confusion_matrix


def train_and_evaluate_linear_nn(X_train_raw, X_test_raw, y_train_raw, y_test_raw):
    X_train, X_test, y_train_onehot, y_test_onehot, y_train, y_test = preprocess_for_nn(X_train_raw, X_test_raw, y_train_raw, y_test_raw)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 10)
    ).to(device)

    n_epochs = 200
    batch_size = 64
    learning_rate = 1e-3
    weight_decay = 1e-4
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_epoch_loss_train = []
    total_epoch_acc_train = []
    total_epoch_loss_test = []
    total_epoch_acc_test = []
    train_dataset = TensorDataset(X_train, y_train_onehot)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataset = TensorDataset(X_test, y_test_onehot)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    for n in range(n_epochs):
        print(f'Epoch {n}')
        model.train()
        epoch_loss = 0
        epoch_acc = 0
        for i, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            # calculate metrics
            epoch_loss += loss.item() * X.size(0)
            epoch_acc += (pred.argmax(dim=1) == y.argmax(dim=1)).sum()
            # backpropagation
            optimizer.zero_grad()  # zero previous gradients
            loss.backward()  # calculate gradients
            optimizer.step()  # update weights
        total_epoch_loss_train.append(epoch_loss / len(train_dataset))
        total_epoch_acc_train.append(epoch_acc / len(train_dataset))
        model.eval()
        with torch.no_grad():
            test_loss = 0
            test_acc = 0
            for i, (X, y) in enumerate(test_dataloader):
                X, y = X.to(device), y.to(device)
                pred = model(X)
                loss = loss_fn(pred, y)
                test_loss += loss.item() * X.size(0)
                test_acc += (pred.argmax(dim=1) == y.argmax(dim=1)).sum()
            total_epoch_loss_test.append(test_loss / len(test_dataset))
            total_epoch_acc_test.append(test_acc / len(test_dataset))

    plot_training_curves(total_epoch_loss_train, total_epoch_loss_test, total_epoch_acc_train, total_epoch_acc_test)

    model.eval()
    y_pred = []
    with torch.no_grad():
        for X, y in test_dataloader:
            X = X.to(device)
            pred = model(X)
            y_pred.append(pred.argmax(dim=1))
    y_pred = torch.cat(y_pred).cpu().numpy()

    plot_confusion_matrix(y_test, y_pred)
