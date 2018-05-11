"""Evaluate the model"""

import logging
import os

import numpy as np
import torch
import utils
import model.net as net
# from model.data_loader import DataLoader


def evaluate(model, loss_fn, data_dict, metrics, data_loader, params, num_steps):
    """Evaluate the model on `num_steps` batches.
    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        data_iterator: (generator) a generator that generates batches of data and labels
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []
    
    data_iterator = data_loader.data_iterator(data_dict, params)

    # compute metrics over the dataset
    for _ in range(num_steps):
        # fetch the next evaluation batch
        data_batch, labels_batch,_ = next(data_iterator)
        
        # compute model output
        output_batch = model(data_batch)
        loss = loss_fn(output_batch, labels_batch)

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = output_batch.data.cpu().numpy()
        labels_batch = labels_batch.data.cpu().numpy()

        # compute all metrics on this batch
        summary_batch = {metric: metrics[metric](output_batch, labels_batch, params)
                         for metric in metrics}
        summary_batch['loss'] = loss.item()
        summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]} 
    f1 = f_score_simple(model, data_dict, data_loader, params, num_steps)
    metrics_mean['f1'] = f1
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean



def f_score_simple(model, data_dict, data_loader, params, num_steps):
    """
    Compute the accuracy, given the outputs and labels for all tokens. Exclude PADding terms.
    Args:
        outputs: (np.ndarray) dimension batch_size*seq_len x num_tags - log softmax output of the model
        labels: (np.ndarray) dimension batch_size x seq_len where each element is either a label in
                [0, 1, ... num_tag-1], or -1 in case it is a PADding token.
    Returns: (float) f1 score in [0,100]
    """
    
    data_iterator = data_loader.data_iterator(data_dict, params)

    correct_preds = 0
    total_correct = 0
    total_predict = 0
    
    for _ in range(num_steps):
        
        data_batch, labels_batch,_ = next(data_iterator)
        
        # compute model output
        output_batch = model(data_batch)

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = output_batch.data.cpu().numpy()
        labels_batch = labels_batch.data.cpu().numpy()
   
        # reshape labels to give a flat vector of length batch_size*seq_len
        labels = labels_batch.ravel()

        # since PADding tokens have label -1, we can generate a mask to exclude the loss from those terms
        mask = (labels >= 0)

        # np.argmax gives us the class predicted for each token by the model
        outputs = np.argmax(output_batch, axis=1)

        O_ind = params.pad_tag_ind # tag_map['O'] == 6

        total_correct += np.sum(mask & (labels != O_ind))
        total_predict += np.sum(outputs != O_ind)
        correct_preds += np.sum((labels == outputs) & (outputs != O_ind))
        
        assert correct_preds <= total_correct, 'correct prediction can not be greater than total entities in labels'
        
    p   = correct_preds / total_predict if correct_preds > 0 else 0
    r   = correct_preds / total_correct if correct_preds > 0 else 0
    f1  = 200 * p * r / (p + r) if correct_preds > 0 else 0
    
    # logging.info('F1 score: ' + str(f1))
    return f1

if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    data_dir = 'data/coNLL/eng/'
    model_dir = 'experiments/coNLL/base_model/'
    json_path = os.path.join(model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()     # use GPU is available

    # Set the random seed for reproducible experiments
    # torch.manual_seed(230)
    if params.cuda: torch.cuda.manual_seed(230)
        
    # Get the logger
    utils.set_logger(os.path.join(model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # load data
    # DataLoader() shouldn't initialized more than once, ideally
    # data_loader = DataLoader(data_dir, params)
    data = data_loader.load_data(['test'], data_dir)
    test_data = data['test']

    test_data_iterator = data_loader.data_iterator(test_data, params)

    
    # Define the model
    model = net.Net(params).cuda() if params.cuda else net.Net(params)
    
    loss_fn = net.loss_fn
    metrics = net.metrics
    
    logging.info("Starting evaluation")

    # Reload weights from the saved file
    restore_file = 'best'
    utils.load_checkpoint(os.path.join(model_dir, restore_file + '.pth.tar'), model)

    num_steps = (params.test_size + 1) // params.batch_size

    test_metrics = evaluate(model, loss_fn, test_data_iterator, metrics, data_loader, params, num_steps)
    
    f_score_simple(model, test_data_iterator, data_loader, params, num_steps)
