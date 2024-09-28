import torch
from torch import nn
from pathlib import Path
from .helper_functions import accuracy_fn
from .preprocess import X_test, X_train, y_test, y_train, device


# Making the Model
class SpamClassifierV0(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=34116, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)

    def forward(self, x):
        return self.layer_3(self.layer_2((self.layer_1(x))))


def train(X_train, X_test, y_train, y_test, device):
    # Calling the model
    model_0 = SpamClassifierV0().to(device)

    # Defining the loss function and the optimizer
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.01)

    # Training the Model
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    epochs = 100

    X_train, X_test = X_train.to(device), X_test.to(device)
    y_train, y_test = y_train.to(device), y_test.to(device)

    for epoch in range(epochs):
        model_0.train()
        y_logits = model_0(X_train).squeeze()
        y_pred = torch.round(torch.sigmoid(y_logits))
        loss = loss_fn(y_logits, y_train)
        acc = accuracy_fn(y_true=y_train, y_pred=y_pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Opening in eval mode for test prediction
        model_0.eval()

        with torch.inference_mode():
            test_logits = model_0(X_test).squeeze()
            test_pred = torch.round(torch.sigmoid(test_logits))
            test_loss = loss_fn(test_logits, y_test)
            test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)

        if epoch % 10 == 0:
            print(
                f"Epoch: {epoch} | Loss: {loss:.5f} | Acc: {acc:.2f}% | "
                f"Test Loss: {test_loss:.5f} | Test Acc : {test_acc:0.2f}")

    # Saving the weights of the model file
    MODEL_PATH = Path("../models/spam_classifier_v0.pth")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj=model_0.state_dict(), f=MODEL_PATH)


if __name__ == "__main__":
    train(X_train, X_test, y_train, y_test, device)
