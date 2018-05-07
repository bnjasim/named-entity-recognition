


# load data
data_loader = DataLoader(data_dir, params)
data = data_loader.load_data(['test'], data_dir)
test_data = data['test']

test_data_iterator = data_loader.data_iterator(test_data, params)

num_steps = (params.test_size + 1) // params.batch_size

def f_score_simple(model, test_data_iterator, params, num_steps):
    """
    Compute the accuracy, given the outputs and labels for all tokens. Exclude PADding terms.
    Args:
        outputs: (np.ndarray) dimension batch_size*seq_len x num_tags - log softmax output of the model
        labels: (np.ndarray) dimension batch_size x seq_len where each element is either a label in
                [0, 1, ... num_tag-1], or -1 in case it is a PADding token.
    Returns: (float) f1 score in [0,100]
    """
    
    correct_preds = 0
    total_correct = 0
    total_predict = 0
    
    for _ in range(num_steps):
        
        data_batch, labels_batch = next(test_data_iterator)
        
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
    f1  = 2 * p * r / (p + r) if correct_preds > 0 else 0
    
    print('F1 score: ' + str(f1))