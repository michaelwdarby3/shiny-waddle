import torch
import torch_geometric
import itertools

from torch_geometric.loader import DataLoader
from torch.nn import Module, CrossEntropyLoss, NLLLoss, MSELoss
from torch_geometric.datasets import FakeDataset
from sklearn.model_selection import KFold
from torch_geometric.nn import SAGEConv, GCNConv
from torch.optim import Optimizer
from torch.nn.functional import nll_loss, relu, dropout, log_softmax

device = "cuda" if torch.cuda.is_available() else "cpu"

class GNN(Module):
    """
    The simple Graphical Neural Net we're working with.
    """
    def __init__(self, hidden_channels, out_channels):
        """
        :param hidden_channels: The size of the inner layer the data passing through is compressed to.
        :param out_channels: The size of the output.
        """
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def __repr__(self):
        return "GNN"

    def forward(self, x, edge_index):
        """
        The forward pass that data gets run through.
        :param x: The data being passed through
        :param edge_index: the tensor defining the source and target nodes of all edges
        :return: the transformed graph.
        """
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)

        x = torch.sum(x.transpose(0,1), dim=1)
        x = torch.reshape(x, (1, 10))

        return x


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        self.num_node_features=64
        self.num_classes=10
        super().__init__()
        self.conv1 = GCNConv(self.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def __repr__(self):
        return "GCN"

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = relu(x)
        x = dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = log_softmax(x, dim=1)

        x = torch.sum(x.transpose(0, 1), dim=1) / x.size()[0]
        x = torch.reshape(x, (1, 10))

        return x


def reset_weights(m: Module):
    """
    Helper function to reset weights of a given model and avoid weight bleed.
    :param m: The model whose weights are being reset.
    :return: None
    """
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            #print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()


def evaluation_model(model: Module, test_loader: DataLoader, f: int):
    """
    Evaluates a model after training.
    :param model: A fully trained GNN instance.
    :param test_loader: A DataLoader object that will give us the test data.
    :param f: The number of fold we're on.
    :return: The results for a given evaluation step.
    """
    correct, total = 0, 0
    results = {}
    with torch.no_grad():
        # Iterate over the test data and generate predictions
        for i, data in enumerate(test_loader, 0):
            # Get inputs
            X = data.x.to(device)
            y = data.y.to(device)
            edge_index = data.edge_index.to(device)

            # Generate outputs
            outputs = model(X, edge_index)

            # Set total and correct
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

        # Print accuracy
        print('Accuracy for fold %d: %d %%' % (f, 100.0 * correct / total))
        print('--------------------------------')
        results[f] = 100.0 * (correct / total)
    return results


def train_model(model: Module, optimizer: Optimizer, loss_function: Module, train_loader: DataLoader, batch_size: int):
    """
    Trains a a given GNN in-place with the data provided.
    :param model: The GNN to be trained.
    :param optimizer: The optimization algorithm object, in this case Adam.
    :param loss_function: Whichever loss function the model is paired with.
    :param train_loader: The DataLoader object that we can get training data from
    :param batch_size: How many units of data are in each batch. Important to keep the same as the number of graphs.
    :return: None.
    """
    current_loss = 0

    model.train()
    for i, data in enumerate(train_loader, 0):

        # Get inputs
        X = data.x.to(device)
        y = data.y.to(device)
        edge_index = data.edge_index.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Perform forward pass
        outputs = model(X, edge_index)

        loss = loss_function(outputs, y)

        # Perform backward pass
        loss.backward()

        # Perform optimization
        optimizer.step()

        # Print statistics
        current_loss += loss.item()
        if i % 500 == 499:
            print('Loss after mini-batch %5d: %.3f' %
                  (i + 1, current_loss / 500))
            current_loss = 0.0


def training_loop(model: Module, num_epochs: int, optimizer: Optimizer, loss_function: Module, train_loader: DataLoader, batch_size: int):
    """
    Handles the training process in its entirety for a given model.
    :param model: The GNN being trained.
    :param num_epochs: How many epochs are being used in this training process
    :param optimizer: The optimization algorithm object being used, in this case Adam.
    :param loss_function: Whichever loss function the model is paired with.
    :param train_loader: The DataLoader object that we can get training data from
    :param batch_size: How many units of data are in each batch. Important to keep the same as the number of graphs.
    :return: None
    """

    for epoch in range(0, num_epochs):

        # Print epoch
        print(f'Starting epoch {epoch + 1}')

        # Iterate over the DataLoader for training data
        train_model(model, optimizer, loss_function, train_loader, batch_size)


def main():
    """
    The outermost loop that performs nested cross validation.
    :return: None
    """

    # The FakeDataset object produces a number of graph objects. I'm keeping the number of graphs to 5 because, for
    # reasons I haven't pinned down, the training process is unstable, as the batch count seems to vary. As a POC,
    # this does the job despite that common bug.
    dataset = FakeDataset(num_graphs=100, num_classes=10)

    num_epochs = 5
    n_splits = 5
    batch_size=5

    # Add or remove hyperparameters from here; it'll dynamically update
    hyperparams = {
        "model": [GNN, GCN],
        "loss_function": [CrossEntropyLoss, NLLLoss],
        "learning_rate": [1e-3, 1e-4, 1e-5]
    }

    nested_results = []

    # This uses itertools to get all combinations of the hyperparameters; this list is 12 long at the moment.
    hyperparam_permutations = [i for i in itertools.product(*hyperparams.values())]

    for i, (model_architecture, loss_function_class, learning_rate) in enumerate(hyperparam_permutations):

        s = f'{str(hyperparam_permutations[i])}'
        print(f'Current Hyperparameters: {s}')

        loss_function = loss_function_class()
        # Creates a number of folds we'll be iterating through.
        validator = KFold(n_splits=n_splits, shuffle=True)

        results = {}

        for f, (trainids, testids) in enumerate(validator.split(dataset)):


            # Sample elements randomly from a given list of ids, no replacement.
            train_dataset = dataset.index_select(trainids)
            test_dataset = dataset.index_select(testids)

            # Define data loaders for training and testing data in this fold
            train_loader = DataLoader(train_dataset)
            test_loader = DataLoader(test_dataset)

            model = model_architecture(hidden_channels=64, out_channels=dataset.num_classes).to(device)

            model.apply(reset_weights)

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            # Runs the actual training process for a given model
            training_loop(model, num_epochs, optimizer, loss_function, train_loader, batch_size)

            # This will save as many models as you train throughout the program for reuse later if you wish.
            #save_path = f'./model-fold-{f}.pth'
            #torch.save(model.state_dict(), save_path)

            # Stores results of evaluation.
            results.update(evaluation_model(model, test_loader, f))

            # Print fold results
        print(f'K-FOLD CROSS VALIDATION RESULTS FOR {n_splits} FOLDS')
        print('--------------------------------')
        sumval = 0.0
        for key, value in results.items():
            print(f'Fold {key}: {value} %')
            sumval += value
        av_val = sumval / len(results.items())
        print(f'Average: {av_val} %')

        nested_results.append(av_val)

    print(f'K*I-FOLD NESTED CROSS VALIDATION RESULTS ACROSS MODEL, LOSS, AND LEARNING RATE')
    print('--------------------------------')
    sumval = 0.0
    best = 0.0
    best_index = -1
    for key, value in enumerate(nested_results):
        s = f'{str(hyperparam_permutations[key])}'
        print(f'Hyperparams {s}: {value} %')
        sumval += value
        if value > best:
            best = value
            best_index = key
    av_val = sumval / len(nested_results)
    print(f'Average: {av_val} %')
    best_hypers = f'{str(hyperparam_permutations[best_index])}'
    print(f'Best hyperparameters: {best_hypers}')
    print(f'Best loss: {best}')




# Runs the script.
if __name__ == '__main__':
    main()
