import os
import importlib
os.environ["DATA_FOLDER"] = "./"
import argparse
import matplotlib.pyplot as plt
import torch
import torch.utils.data
import torch.nn as nn
import random
from utils.parser import *
from utils import datasets
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
import os
import importlib
os.environ["DATA_FOLDER"] = "./"
from sklearn.model_selection import KFold
import argparse
import time
import joblib
import torch
import torch.utils.data
import torch.nn as nn
import matplotlib.pyplot as plt
import networkx as nx
from utils.parser import *
from utils import datasets
from sklearn.metrics import accuracy_score, hamming_loss, f1_score, jaccard_score, precision_score, recall_score
from sklearn.metrics import  average_precision_score, precision_recall_curve, roc_auc_score, auc
import numpy as np
from itertools import combinations


    
def compute_f1_by_level(y_true, y_pred, levels):
    """
    Compute and print macro and micro F1 scores for each level, excluding the root node (level 0).
    
    Parameters:
    y_true (numpy array): Ground truth binary matrix for the test set.
    y_pred (numpy array): Predicted binary matrix for the test set.
    levels (dict): Dictionary mapping each level to a list of node indices at that level.
    """
    num_columns = y_true.shape[1]  # Total number of columns in y_true and y_pred

    for level in sorted(levels.keys()):
        if level == 0:
            # Skip level 0 (root node)
            continue
            
        # Get nodes for the current level and ensure they are within bounds
        nodes = [node for node in levels[level] if node < num_columns]
        
        # Check if nodes exist for the level after filtering
        if not nodes:
            print(f"Level {level} - No nodes within bounds, skipping.")
            continue
        
        # Select only the columns of y_true and y_pred corresponding to the nodes of this level
        y_true_level = y_true[:, nodes]
        y_pred_level = y_pred[:, nodes]
        
        # Compute macro and micro F1 scores for the current level
        macro_f1 = f1_score(y_true_level, y_pred_level, average='macro', zero_division=0)
        micro_f1 = f1_score(y_true_level, y_pred_level, average='micro', zero_division=0)
        
        # Print the results for the current level
        print(f"Level {level} - Macro F1 Score: {macro_f1:.4f}")
        print(f"Level {level} - Micro F1 Score: {micro_f1:.4f}")

def print_labels_by_level(g):
    
    """ Analyzes the graph and prints labels belonging to each level of the hierarchy.
    The graph `g` has edges directed from descendants to ancestors (reverse hierarchy).
    """

    # Find root nodes (nodes with no outgoing edges, i.e., no ancestors)
    root_nodes = [n for n, d in g.out_degree() if d == 0]
    
    # Dictionary to store labels by levels
    levels = {}
    current_level = 0
    levels[current_level] = root_nodes  # Root nodes are at level 0
    
    # Breadth-First Search (BFS) to assign levels, reverse graph direction
    visited = set(root_nodes)
    queue = list(root_nodes)
    
    while queue:
        next_queue = []
        for node in queue:
            for neighbor in g.predecessors(node):  # Find predecessors (descendants -> ancestors)
                if neighbor not in visited:
                    if current_level + 1 not in levels:
                        levels[current_level + 1] = []
                    levels[current_level + 1].append(neighbor)
                    visited.add(neighbor)
                    next_queue.append(neighbor)
        queue = next_queue
        current_level += 1

    # Print the labels by levels
    for level, labels in levels.items():
        print(f"Level {level}: {labels}")
    
    return levels


def compute_level_3_hops_bidirectional(g, levels):
    """
    Compute the hops (shortest path lengths) between nodes in level 3 of the hierarchy.
    Treat the graph as undirected to ensure path finding between nodes in different branches.
    
    Parameters:
    g (networkx DiGraph): The directed hierarchy graph where edges point from descendants to ancestors.
    levels (dict): Dictionary containing nodes organized by levels.
    
    Returns:
    level_3_hops (dict): Dictionary containing pairwise hops between nodes for level 3.
    """
    
    level_3_hops = {}
    
    # Convert the directed graph to an undirected graph for distance calculation
    g_undirected = g.to_undirected()
    
    # Get the nodes in level 3
    level_3_nodes = levels.get(3, [])
    
    # Generate all unique pairs of level 3 nodes
    for node1, node2 in combinations(level_3_nodes, 2):
        try:
            # Compute the shortest path length between the nodes in the undirected graph
            hops = nx.shortest_path_length(g_undirected, source=node1, target=node2)
        except nx.NetworkXNoPath:
            hops = float('inf')  # If no path exists, assign infinity
        
        # Store the hops for the node pair
        level_3_hops[(node1, node2)] = hops
        level_3_hops[(node2, node1)] = hops  # Symmetric, so add reverse direction as well
    
    return level_3_hops



def calculate_hierarchical_metrics(y_true, y_pred, R):
    """Calculate hierarchical precision, recall, and F1 for macro and micro results."""
    num_instances = len(y_true)
    
    total_intersection = 0
    total_predicted = 0
    total_true = 0
    total_macro_hp, total_macro_hr, total_macro_hf = 0, 0, 0
    
    for i in range(num_instances):
        # Get predicted and true indices with ancestors extended
        predicted_indices = y_pred[i].nonzero(as_tuple=True)[0].tolist()
        true_indices = y_true[i].nonzero(as_tuple=True)[0].tolist()
        
        pred_with_ancestors = extend_with_ancestors(predicted_indices, R)
        true_with_ancestors = extend_with_ancestors(true_indices, R)
        
        # Calculate intersection and totals for micro metrics
        intersection = len(pred_with_ancestors.intersection(true_with_ancestors))
        total_intersection += intersection
        total_predicted += len(pred_with_ancestors)
        total_true += len(true_with_ancestors)
        
        # Calculate hierarchical precision, recall, F1 for current instance (for macro)
        if len(pred_with_ancestors) > 0:
            hp_instance = intersection / len(pred_with_ancestors)
        else:
            hp_instance = 0
        if len(true_with_ancestors) > 0:
            hr_instance = intersection / len(true_with_ancestors)
        else:
            hr_instance = 0
        if hp_instance + hr_instance > 0:
            hf_instance = 2 * (hp_instance * hr_instance) / (hp_instance + hr_instance)
        else:
            hf_instance = 0
        
        total_macro_hp += hp_instance
        total_macro_hr += hr_instance
        total_macro_hf += hf_instance
    
    # Calculate micro metrics
    micro_hp = total_intersection / total_predicted if total_predicted > 0 else 0
    micro_hr = total_intersection / total_true if total_true > 0 else 0
    micro_hf = (2 * micro_hp * micro_hr / (micro_hp + micro_hr)) if (micro_hp + micro_hr) > 0 else 0

    # Calculate macro metrics
    macro_hp = total_macro_hp / num_instances
    macro_hr = total_macro_hr / num_instances
    macro_hf = total_macro_hf / num_instances
    
    return {
        "micro_hp": micro_hp, "micro_hr": micro_hr, "micro_hf": micro_hf,
        "macro_hp": macro_hp, "macro_hr": macro_hr, "macro_hf": macro_hf
    }

def get_last_level_labels(g):
    last_level_labels = []
    for node in g.nodes:
        # Check if the node has no outgoing edges (i.e., it's a leaf node)
        if g.out_degree(node) == 0:
            last_level_labels.append(node)
    return last_level_labels

def calculate_hops_lca(g, R, false_positive, true_positive):
    """
    Calculate the number of hops between a false positive and a true positive using the LCA approach.
    
    g: The graph representing the hierarchy.
    R: Ancestor matrix where R_ij = 1 if i is a descendant of j.
    false_positive: The false positive label index.
    true_positive: The true positive label index.
    
    Returns:
    - Number of hops between false_positive and true_positive via the lowest common ancestor.
    """
    # Get the ancestors of both false_positive and true_positive
    ancestors_false_positive = set(np.where(R[false_positive] == 1)[0])
    ancestors_true_positive = set(np.where(R[true_positive] == 1)[0])

    # Find the common ancestors
    common_ancestors = ancestors_false_positive.intersection(ancestors_true_positive)

    if not common_ancestors:
        # If no common ancestors, the distance is very large (but this shouldn't happen in a valid hierarchy)
        return np.inf

    # Find the lowest common ancestor (LCA) - the deepest node in the hierarchy
    lca = max(common_ancestors, key=lambda ancestor: nx.shortest_path_length(g, source=4, target=ancestor))

    # Calculate hops: from true_positive to LCA and from LCA to false_positive
    hops_true_to_lca = nx.shortest_path_length(g, source=true_positive, target=lca)
    hops_lca_to_false = nx.shortest_path_length(g, source=lca, target=false_positive)

    total_hops = hops_true_to_lca + hops_lca_to_false
    return total_hops

def extend_with_ancestors(predicted_indices, R_2d):
    """Extend the set of labels with their ancestors in the hierarchy."""
    extended_labels = set(predicted_indices)
    for label in predicted_indices:
        extended_labels.update(get_ancestors(label, R_2d))
    return extended_labels


# Function to get ancestors of a label in the hierarchy matrix
def get_ancestors(label_idx, R_2d, root_idx=4):
    """Retrieve all ancestors of a given label from the hierarchy matrix R_2d."""
    ancestors = set()
    ancestor_indices = torch.nonzero(R_2d[0, :, label_idx]).squeeze()
    if ancestor_indices.dim() == 0:  # Single ancestor case
        ancestor_indices = ancestor_indices.unsqueeze(0)

    ancestor_indices = ancestor_indices.tolist()
    ancestors.update([a for a in ancestor_indices if a != root_idx])  # Avoid adding the root

    return ancestors



def get_constr_out(x, R):
    """ Given the output of the neural network x returns the output of MCM given the hierarchy constraint expressed in the matrix R """
    c_out = x.double()
    c_out = c_out.unsqueeze(1)
    c_out = c_out.expand(len(x),R.shape[1], R.shape[1])
    R_batch = R.expand(len(x),R.shape[1], R.shape[1])
    final_out, _ = torch.max(R_batch*c_out.double(), dim = 2)
    return final_out


class ConstrainedFFNNModel(nn.Module):
    """ C-HMCNN(h) model - during training it returns the not-constrained output that is then passed to MCLoss """
    def __init__(self, input_dim, hidden_dim, output_dim, hyperparams, R):
        super(ConstrainedFFNNModel, self).__init__()
        
        self.nb_layers = hyperparams['num_layers']
        self.R = R
        
        fc = []
        for i in range(self.nb_layers):
            if i == 0:
                fc.append(nn.Linear(input_dim, hidden_dim))
            elif i == self.nb_layers-1:
                fc.append(nn.Linear(hidden_dim, output_dim))
            else:
                fc.append(nn.Linear(hidden_dim, hidden_dim))
        self.fc = nn.ModuleList(fc)
        
        self.drop = nn.Dropout(hyperparams['dropout'])
        
        
        self.sigmoid = nn.Sigmoid()
        if hyperparams['non_lin'] == 'tanh':
            self.f = nn.Tanh()
        else:
            self.f = nn.ReLU()
        
    def forward(self, x):
        for i in range(self.nb_layers):
            if i == self.nb_layers-1:
                x = self.sigmoid(self.fc[i](x))
            else:
                x = self.f(self.fc[i](x))
                x = self.drop(x)
        if self.training:
            constrained_out = x
        else:
            constrained_out = get_constr_out(x, self.R)
        return constrained_out



def main():
    
    total_correct_intersection = 0 
    total_predicted_ancestors= 0
    total_true_ancestors= 0
    total_avg_hops = 0  # To track the average hops across all test examples
    num_instances = 0


    parser = argparse.ArgumentParser(description='Train neural network on train and validation set')

    # Required  parameter
    parser.add_argument('--dataset', type=str, default=None, required=True,
                        help='dataset name, must end with: "_GO", "_FUN", or "_others"' )
    # Other parameters
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 0)')
    parser.add_argument('--device', type=str, default='0',
                        help='GPU (default:0)')
    args = parser.parse_args()

    assert('_' in args.dataset)
    assert('FUN' in args.dataset or 'GO' in args.dataset or 'others' in args.dataset)

    # Load train, val and test set
    dataset_name = args.dataset
    data = dataset_name.split('_')[0]
    ontology = dataset_name.split('_')[1]

    # Dictionaries with number of features and number of labels for each dataset
    input_dims = {'originaldefects':482, 'finaldefects':486}
    output_dims_others = {'originaldefects':55,'finaldefects':55}
    output_dims = {'others':output_dims_others}

    #Dictionaries with the hyperparameters associated to each dataset
    hidden_dims_others = {'originaldefects':1000,'finaldefects':1000}
    hidden_dims = {'others':hidden_dims_others}
    lrs_others = {'originaldefects':1e-5,'finaldefects':1e-5}
    lrs = {'others':lrs_others}
    epochss_others = {'originaldefects':100,'finaldefects':2}
    epochss = { 'others':epochss_others}

    # Set the hyperparameters 
    batch_size = 4
    num_layers = 3
    dropout = 0.7
    non_lin = 'relu'
    hidden_dim = hidden_dims[ontology][data]
    lr = lrs[ontology][data]
    weight_decay = 1e-5
    num_epochs = epochss[ontology][data]
    hyperparams = {'batch_size':batch_size, 'num_layers':num_layers, 'dropout':dropout, 'non_lin':non_lin, 'hidden_dim':hidden_dim, 'lr':lr, 'weight_decay':weight_decay}


    # Set seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available:
        pin_memory = True


    # Load the datasets
    if ('others' in args.dataset):
        train, test = initialize_other_dataset(dataset_name, datasets)
        train.to_eval, test.to_eval = torch.tensor(train.to_eval, dtype=torch.uint8),  torch.tensor(test.to_eval, dtype=torch.uint8)
    else:
        train, val, test = initialize_dataset(dataset_name, datasets)
        train.to_eval, val.to_eval, test.to_eval = torch.tensor(train.to_eval, dtype=torch.uint8), torch.tensor(val.to_eval, dtype=torch.uint8), torch.tensor(test.to_eval, dtype=torch.uint8)
    
    different_from_0 = torch.tensor(np.array((test.Y.sum(0)!=0), dtype = np.uint8), dtype=torch.uint8)

    # Compute matrix of ancestors R
    # Given n classes, R is an (n x n) matrix where R_ij = 1 if class i is descendant of class j
    num_nodes = len(train.A.nodes)
    R = np.zeros((num_nodes, num_nodes))
    np.fill_diagonal(R, 1)
    g = nx.DiGraph(train.A) # train.A is the matrix where the direct connections are stored 
    for i in range(len(train.A)):
        ancestors = list(nx.descendants(g, i)) #here we need to use the function nx.descendants() because in the directed graph the edges have source from the descendant and point towards the ancestor 
        if ancestors:
            R[i, ancestors] = 1
    R = torch.tensor(R)
    #Transpose to get the descendants for each node 
    R = R.transpose(1, 0)
    R = R.unsqueeze(0).to(device)
    levels = print_labels_by_level(g)
    level_3_hops = compute_level_3_hops_bidirectional(g, levels)
    levels = print_labels_by_level(g)
    level_3_nodes = levels.get(3, [])  # Extract level 3 nodes here





    # Rescale data and impute missing data
    if ('others' in args.dataset):
        scaler = preprocessing.StandardScaler().fit((train.X.astype(float)))
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean').fit((train.X.astype(float)))
    else:
        scaler = preprocessing.StandardScaler().fit(np.concatenate((train.X, val.X)))
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean').fit(np.concatenate((train.X, val.X)))
        val.X, val.Y = torch.tensor(scaler.transform(imp_mean.transform(val.X))).to(device), torch.tensor(val.Y).to(device)
    train.X, train.Y = torch.tensor(scaler.transform(imp_mean.transform(train.X))).to(device), torch.tensor(train.Y).to(device)        
    test.X, test.Y = torch.tensor(scaler.transform(imp_mean.transform(test.X))).to(device), torch.tensor(test.Y).to(device)

    #Create loaders 
    train_dataset = [(x, y) for (x, y) in zip(train.X, train.Y)]
    if ('others' not in args.dataset):
        val_dataset = [(x, y) for (x, y) in zip(val.X, val.Y)]
        for (x, y) in zip(val.X, val.Y):
            train_dataset.append((x,y))
    test_dataset = [(x, y) for (x, y) in zip(test.X, test.Y)]

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=False)

    # We do not evaluate the performance of the model on the 'roots' node (https://dtai.cs.kuleuven.be/clus/hmcdatasets/)
    if 'GO' in dataset_name: 
        num_to_skip = 4
    else:
        num_to_skip = 1 

    # Create the model
    model = ConstrainedFFNNModel(input_dims[data], hidden_dim, output_dims[ontology][data]+num_to_skip, hyperparams, R)
    model.to(device)
    print("Model on gpu", next(model.parameters()).is_cuda)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCELoss()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, (x, labels) in enumerate(train_loader):

            x = x.to(device)
            labels = labels.to(device)
        
            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()
            output = model(x.float())

            #MCLoss
            constr_output = get_constr_out(output, R)
            train_output = labels*output.double()
            train_output = get_constr_out(train_output, R)
            train_output = (1-labels)*constr_output.double() + labels*train_output

            loss = criterion(train_output[:,train.to_eval], labels[:,train.to_eval]) 

            predicted = constr_output.data > 0.5

            # Total number of labels
            total_train = labels.size(0) * labels.size(1)
            # Total correct predictions
            correct_train = (predicted == labels.byte()).sum()

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}')

    for i, (x,y) in enumerate(test_loader):
        
        model.eval()
                
        x = x.to(device)
        y = y.to(device)

        constrained_output = model(x.float())
        predicted = constrained_output.data > 0.5
        # Total number of labels
        total = y.size(0) * y.size(1)
        # Total correct predictions
        correct = (predicted == y.byte()).sum()

        #Move output and label back to cpu to be processed by sklearn
        predicted = predicted.to('cpu')
        cpu_constrained_output = constrained_output.to('cpu')
        y = y.to('cpu')

        if i == 0:
            predicted_test = predicted
            constr_test = cpu_constrained_output
            y_test = y
        else:
            predicted_test = torch.cat((predicted_test, predicted), dim=0)
            constr_test = torch.cat((constr_test, cpu_constrained_output), dim=0)
            y_test = torch.cat((y_test, y), dim =0)

        predicted_indices = predicted.nonzero(as_tuple=True)[1].tolist()  # assuming binary labels
        true_indices = y.nonzero(as_tuple=True)[1].tolist()

        pred_with_ancestors = extend_with_ancestors(predicted_indices, R)
        true_with_ancestors = extend_with_ancestors(true_indices, R)
        correct_intersection = len(pred_with_ancestors.intersection(true_with_ancestors))
        total_predicted = len(pred_with_ancestors)
        total_true = len(true_with_ancestors)

        total_correct_intersection += correct_intersection
        total_predicted_ancestors += total_predicted
        total_true_ancestors += total_true

        false_positives = set(predicted_indices) - set(true_indices)
        last_level_labels = get_last_level_labels(g)

        hops_sum = 0
        count_false_positives_last_level = 0

        for false_positive in false_positives:
           if false_positive in last_level_labels:
              min_hops = np.inf  # Start with a high value to find the minimum hops
              for true_positive in true_indices:
                  hops = calculate_hops_lca(g, R, false_positive, true_positive)
                  if hops < min_hops:
                    min_hops = hops
              hops_sum += min_hops
              count_false_positives_last_level += 1

        if count_false_positives_last_level > 0:
           avg_hops = hops_sum / count_false_positives_last_level
           #print(f"Average hops for false positives in the last level: {avg_hops}")
           total_avg_hops += avg_hops

        num_instances += 1

    if num_instances > 0:
       overall_avg_hops = total_avg_hops / num_instances
       print(f"Overall average hops across test instances: {overall_avg_hops}")



    hP = total_correct_intersection / total_predicted_ancestors if total_predicted_ancestors > 0 else 0
    hR = total_correct_intersection / total_true_ancestors if total_true_ancestors > 0 else 0

    # Hierarchical F1 score
    hF = (2 * hP * hR) / (hP + hR) if (hP + hR) > 0 else 0

    print(f'Hierarchical Precision: {hP}')
    print(f'Hierarchical Recall: {hR}')
    print(f'Hierarchical F1 Score: {hF}')
    compute_f1_by_level(y_test[:, test.to_eval], predicted_test[:, test.to_eval], levels)
    score = average_precision_score(y_test[:,test.to_eval], constr_test.data[:,test.to_eval], average='micro')
    accuracy = accuracy_score(y_test[:,test.to_eval], predicted_test[:,test.to_eval])
    hamming = hamming_loss(y_test[:,test.to_eval], predicted_test[:,test.to_eval])
    f1 = f1_score(y_test[:,test.to_eval], predicted_test[:,test.to_eval], average='micro')
    f1_macro = f1_score(y_test[:, test.to_eval], predicted_test[:, test.to_eval], average='macro') 
    jaccard = jaccard_score(y_test[:,test.to_eval], predicted_test[:,test.to_eval], average='samples')
    avg_precision = average_precision_score(y_test[:,test.to_eval], constr_test.data[:,test.to_eval], average='micro')
    roc_auc = roc_auc_score(y_test[:,test.to_eval], constr_test.data[:,test.to_eval], average='micro')
    precision_micro = precision_score(y_test[:, test.to_eval], predicted_test[:, test.to_eval], average='micro')
    precision_macro = precision_score(y_test[:, test.to_eval], predicted_test[:, test.to_eval], average='macro')
    recall_micro = recall_score(y_test[:, test.to_eval], predicted_test[:, test.to_eval], average='micro')
    recall_macro = recall_score(y_test[:, test.to_eval], predicted_test[:, test.to_eval], average='macro')
    print(f"Micro-averaged ROC AUC: {roc_auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Hamming Loss: {hamming:.4f}")
    print(f"Micro F1 Score: {f1:.4f}")
    print(f"Macro F1 Score: {f1_macro:.4f}")
    print(f"Jaccard Score: {jaccard:.4f}")
    print(f"Average Precision Score: {avg_precision:.4f}")
    print(f"Micro Precision: {precision_micro:.4f}")  
    print(f"Macro Precision: {precision_macro:.4f}")  
    print(f"Micro Recall: {recall_micro:.4f}")  
    print(f"Macro Recall: {recall_macro:.4f}")  
    results = calculate_hierarchical_metrics(y_test, predicted_test, R)
    print("Micro HP:", results["micro_hp"])
    print("Micro HR:", results["micro_hr"])
    print("Micro HF:", results["micro_hf"])
    print("Macro HP:", results["macro_hp"])
    print("Macro HR:", results["macro_hr"])
    print("Macro HF:", results["macro_hf"])


    f = open('results/'+dataset_name+'.csv', 'a')
    f.write(str(seed)+ ',' +str(epoch) + ',' + str(score) + '\n')
    f.close()

if __name__ == "__main__":
    main()