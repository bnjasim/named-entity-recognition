
# coding: utf-8

# In[1]:


import logging
import os

import numpy as np
import torch
import torch.optim as optim
from tqdm import trange
from tqdm import tqdm_notebook as tqdm

import utils
import model.net as net
from model.data_loader import DataLoader
from evaluate import evaluate, f_score_simple


# In[2]:


# data_dir = 'data/coNLL/eng/'
# model_dir = 'experiments/coNLL/base_model/'
data_dir = 'data/kaggle/'
model_dir = 'experiments/kaggle/base_model/'
json_path = os.path.join(model_dir, 'params.json')
params = utils.Params(json_path)


# In[3]:


# use GPU if available
params.cuda = torch.cuda.is_available()
params.dict


# In[4]:


# load data
data_loader = DataLoader(data_dir, params)
data = data_loader.load_data(['train', 'val', 'test'])
train_data = data['train']
val_data = data['val']
test_data = data['test']

# specify the train and val dataset sizes
params.train_size = train_data['size']
params.val_size = val_data['size']
params.test_size = test_data['size']

params.pad_tag_ind = data_loader.tag_map[params.pad_tag]


# In[5]:


data_loader.dataset_params.dict


# In[6]:


# specify the train and val dataset sizes
params.train_size = train_data['size']
params.val_size = val_data['size']
params.pad_tag_ind = data_loader.tag_map[params.pad_tag]


# In[7]:


# Define the model and optimizer
model = net.Net(params).cuda()


# In[8]:


optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
# fetch loss function and metrics
loss_fn = net.loss_fn
metrics = net.metrics


# In[9]:


# Set the logger
utils.set_logger(os.path.join(model_dir, 'train.log'))


# In[10]:


def train(model, optimizer, loss_fn, data_iterator, metrics, params, num_steps):
    """Train the model on `num_steps` batches
    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        data_iterator: (generator) a generator that generates batches of data and labels
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()
    
    # Use tqdm for progress bar
    t = trange(num_steps) 
    for i in t:
        # fetch the next training batch
        train_batch, labels_batch = next(data_iterator)

        # compute model output and loss
        output_batch = model(train_batch)
        loss = loss_fn(output_batch, labels_batch)

        # clear previous gradients, compute gradients of all variables wrt loss
        optimizer.zero_grad()
        loss.backward()

        # performs updates using calculated gradients
        optimizer.step()

        # Evaluate summaries only once in a while
        if i % params.save_summary_steps == 0:
            # extract data from torch Variable, move to cpu, convert to numpy arrays
            output_batch = output_batch.data.cpu().numpy()
            labels_batch = labels_batch.data.cpu().numpy()

            # compute all metrics on this batch
            summary_batch = {metric:metrics[metric](output_batch, labels_batch, params) for metric in metrics}
            summary_batch['loss'] = loss.item()
            summ.append(summary_batch)
            
            # print('Evaluate called')
            
            

        # update the average loss
        loss_avg.update(loss.item())
        t.set_postfix(loss='{:05.3f}'.format(loss_avg()))

    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]} 
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)


# In[11]:


best_val_acc = 0.0

for epoch in range(params.num_epochs):
    # Run one epoch
    logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

    # compute number of batches in one epoch (one full pass over the training set)
    num_steps = (params.train_size + 1) // params.batch_size
    train_data_iterator = data_loader.data_iterator(train_data, params, shuffle=True)
    train(model, optimizer, loss_fn, train_data_iterator, metrics, params, num_steps)

    # Evaluate for one epoch on validation set
    num_steps = (params.val_size + 1) // params.batch_size
    val_data_iterator = data_loader.data_iterator(val_data, params, shuffle=False)
    val_metrics = evaluate(model, loss_fn, val_data, metrics, data_loader, params, num_steps)

    val_acc = val_metrics['accuracy']
    is_best = val_acc >= best_val_acc

    # Save weights
    utils.save_checkpoint({'epoch': epoch + 1,
                           'state_dict': model.state_dict(),
                           'optim_dict' : optimizer.state_dict()}, 
                           is_best=is_best,
                           checkpoint=model_dir)

    # If best_eval, best_save_path        
    if is_best:
        logging.info("- Found new best accuracy")
        best_val_acc = val_acc

        # Save best val metrics in a json file in the model directory
        best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
        utils.save_dict_to_json(val_metrics, best_json_path)

    # Save latest val metrics in a json file in the model directory
    last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
    utils.save_dict_to_json(val_metrics, last_json_path)


# In[12]:


num_steps = (params.val_size + 1) // params.batch_size
f_score_simple(model, val_data, data_loader, params, num_steps)


# ## Evaluate

# In[14]:


# Define the model
model = net.Net(params).cuda() if params.cuda else net.Net(params)

loss_fn = net.loss_fn
metrics = net.metrics

logging.info("Starting evaluation")

restore_file = 'best'
# Reload weights from the saved file
r = utils.load_checkpoint(os.path.join(model_dir, restore_file + '.pth.tar'), model)


# In[15]:


# Evaluate
num_steps = (params.test_size + 1) // params.batch_size
test_metrics = evaluate(model, loss_fn, test_data, metrics, data_loader, params, num_steps)

