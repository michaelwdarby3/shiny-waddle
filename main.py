import torch
import torch_geometric

from torch.utils.data import DataLoader
from torch import nn
from torch.nn import Module
from torch_geometric.loader import NeighborLoader
from torch_geometric.datasets import FakeDataset
from sklearn.model_selection import KFold
from torch_geometric.nn import SAGEConv
from torch.optim import Optimizer


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

    def forward(self, x, edge_index):
        """
        The forward pass that data gets run through.
        :param x: The data being passed through
        :param edge_index: the tensor defining the source and target nodes of all edges
        :return: the transformed graph.
        """
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


def reset_weights(m: Module):
    """
    Helper function to reset weights of a given model and avoid weight bleed.
    :param m: The model whose weights are being reset.
    :return: None
    """
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()


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
    for i, data in enumerate(train_loader, 0):

        # Get inputs
        inputs = data.x
        targets = data.y
        edge_index = data.edge_index
        if inputs.size()[0] != batch_size:
            continue

        # Zero the gradients
        optimizer.zero_grad()

        # Perform forward pass
        outputs = model(inputs, edge_index)

        loss = loss_function(outputs, targets)

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
            inputs = data.x
            targets = data.y
            edge_index = data.edge_index

            # Generate outputs
            outputs = model(inputs, edge_index)

            # Set total and correct
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        # Print accuracy
        print('Accuracy for fold %d: %d %%' % (f, 100.0 * correct / total))
        print('--------------------------------')
        results[f] = 100.0 * (correct / total)
    return results


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


def run_loop():
    """
    The outermost loop that performs nested cross validation.
    :return: None
    """

    # The FakeDataset object produces a number of graph objects. I'm keeping the number of graphs to 5 because, for
    # reasons I haven't pinned down, the training process is unstable, as the batch count seems to vary. As a POC,
    # this does the job despite that common bug.
    dataset = FakeDataset(num_graphs=5)

    batch_size = 5
    num_epochs = 5
    n_splits = 5
    results = {}
    loss_function = nn.CrossEntropyLoss()

    # Creates a number of folds we'll be iterating through.
    validator = KFold(n_splits=n_splits, shuffle=True)

    for f, (trainids, testids) in enumerate(validator.split(dataset)):

        # Sample elements randomly from a given list of ids, no replacement.
        train_dataset = dataset.index_select(trainids)
        test_dataset = dataset.index_select(testids)

        # We use the NeighborSampler here because it's a relatively straightforward way to iterate through a graph.
        train_subsampler = torch_geometric.sampler.NeighborSampler(dataset.data, num_neighbors=[0])
        test_subsampler = torch_geometric.sampler.NeighborSampler(dataset.data, num_neighbors=[0])

        # Define data loaders for training and testing data in this fold
        train_loader = NeighborLoader(dataset.data, num_neighbors=[5], neighbor_sampler=train_subsampler, batch_size=batch_size, shuffle=True)
        test_loader = NeighborLoader(dataset.data, num_neighbors=[5], neighbor_sampler=test_subsampler, batch_size=batch_size, shuffle=True)

        model = GNN(hidden_channels=64, out_channels=dataset.num_classes)

        model.apply(reset_weights)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Runs the actual training process for a given model
        training_loop(model, num_epochs, optimizer, loss_function, train_loader, batch_size)

        # This will save as many models as you train throughout the program for reuse later if you wish.
        save_path = f'./model-fold-{f}.pth'
        torch.save(model.state_dict(), save_path)

        # Stores results of evaluation.
        results.update(evaluation_model(model, test_loader, f))

        # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {n_splits} FOLDS')
    print('--------------------------------')
    sumval = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value} %')
        sumval += value
    print(f'Average: {sumval / len(results.items())} %')

# Runs the script.
if __name__ == '__main__':
    run_loop()

