import random
import numpy as np
import os
import sys

import torch
from torch.autograd import Variable

import utils 


class DataLoader(object):
    """
    Handles all aspects of the data. Stores the dataset_params, vocabulary and tags with their mappings to indices.
    """
    def __init__(self, data_dir, params, path_glove=''):
        """
        Loads dataset_params, vocabulary and tags. Ensure you have run `build_vocab.py` on data_dir before using this
        class.

        Args:
            data_dir: (string) directory containing the dataset
            params: (Params) hyperparameters of the training process. This function modifies params and appends
                    dataset_params (such as vocab size, num_of_tags etc.) to params.
        """
        self.data_dir = data_dir
        # loading dataset_params
        json_path = os.path.join(data_dir, 'dataset_params.json')
        assert os.path.isfile(json_path), "No json file found at {}, run build_vocab.py".format(json_path)
        self.dataset_params = utils.Params(json_path)        
        
        # loading vocab (we require this to map words to their indices)
        vocab_path = os.path.join(data_dir, 'words.txt')
        self.vocab = {}
        self.vocab_lower = {} # only keep lower case words
        self.idx_vocab = {}
        self.idx_vocab_lower = {}
        with open(vocab_path) as f:
            c = 0
            for i, l in enumerate(f.read().splitlines()):
                # l = self.unicodeToAscii(l)
                self.vocab[l] = i
                self.idx_vocab[i] = l
                
                w = l.lower()
                if w not in self.vocab_lower:
                    self.vocab_lower[w] = c
                    self.idx_vocab_lower[c] = w
                    c += 1   
 
        # setting the indices for UNKnown words and PADding symbols
        self.unk_ind = self.vocab[self.dataset_params.unk_word]
        self.pad_ind = self.vocab[self.dataset_params.pad_word]
                
        # loading tags (we require this to map tags to their indices)
        tags_path = os.path.join(data_dir, 'tags.txt')
        self.tag_map = {}
        self.idx_tag = {}
        with open(tags_path) as f:
            for i, t in enumerate(f.read().splitlines()):
                self.tag_map[t] = i
                self.idx_tag[i] = t

        # Glove embeddings
        glove_dict = {}

        if params.use_glove and path_glove:
            # glove_vocab = set()
            with open(path_glove) as f:
                for line in f:
                    s = line.strip().split(' ')
                    word = s[0]
                    # glove_vocab.add(word)
                    if word in self.vocab_lower:
                        emb = [float(j) for j in s[1:]]
                        glove_dict[word] = emb
        
        self.embedding = np.zeros((len(self.vocab), params.embedding_dim))
        # Embedding Initialization
        # If word not present in glove, then initialize randomly
        with open(vocab_path) as f:
            for i, word in enumerate(f.read().splitlines()):
                word = word.lower()
                if glove_dict.get(word):
                    self.embedding[i] = glove_dict[word]
                else:
                    self.embedding[i] = np.random.randn(params.embedding_dim)

        

        # adding dataset parameters to param (e.g. vocab size, )
        params.update(json_path)

        assert params.vocab_size == len(self.vocab), 'Vocabulary sizes from build_vocab and in Data loader should match'

    def load_sentences_labels(self, sentences_file, labels_file, d):
        """
        Loads sentences and labels from their corresponding files. Maps tokens and tags to their indices and stores
        them in the provided dict d.

        Args:
            sentences_file: (string) file with sentences with tokens space-separated
            labels_file: (string) file with NER tags for the sentences in labels_file
            d: (dict) a dictionary in which the loaded data is stored
        """

        sentences = []
        labels = []

        with open(sentences_file, encoding='ISO-8859-1') as f:
            for sentence in f.readlines():
                # replace each token by its index if it is in vocab
                # else use index of UNK_WORD
                sentence = sentence.strip()
                if sentence:
                    # sent_uni = [self.unicodeToAscii(token) for token in sentence.split(' ')]
                    s = [self.vocab[token] if token in self.vocab 
                         else self.unk_ind
                         for token in sentence.split(' ')]
                    sentences.append(s)
        
        with open(labels_file) as f:
            for sentence in f.read().splitlines():
                # replace each label by its index
                l = [self.tag_map[label] for label in sentence.split(' ')]
                labels.append(l)        
        
        print(len(sentences), len(labels))
        # checks to ensure there is a tag for each token
        assert len(labels) == len(sentences)
        for i in range(len(labels)):
            assert len(labels[i]) == len(sentences[i])

        # storing sentences and labels in dict d
        d['data'] = sentences
        d['labels'] = labels
        d['size'] = len(sentences)

    def load_data(self, types):
        """
        Loads the data for each type in types from data_dir.

        Args:
            types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
            data_dir: (string) directory containing the dataset

        Returns:
            data: (dict) contains the data with labels for each type in types

        """
        data = {}
        
        for split in ['train', 'val', 'test']:
            if split in types:
                sentences_file = os.path.join(self.data_dir, split, "sentences.txt")
                labels_file = os.path.join(self.data_dir, split, "labels.txt")
                data[split] = {}
                self.load_sentences_labels(sentences_file, labels_file, data[split])

        return data

    def data_iterator(self, data, params, shuffle=False):
        """
        Returns a generator that yields batches data with labels. Batch size is params.batch_size. Expires after one
        pass over the data.

        Args:
            data: (dict) contains data which has keys 'data', 'labels' and 'size'
            params: (Params) hyperparameters of the training process.
            shuffle: (bool) whether the data should be shuffled

        Yields:
            batch_data: (Variable) dimension batch_size x seq_len with the sentence data
            batch_labels: (Variable) dimension batch_size x seq_len with the corresponding labels

        """

        # make a list that decides the order in which we go over the data- this avoids explicit shuffling of data
        order = list(range(data['size']))
        if shuffle:
            # random.seed(230)
            random.shuffle(order)

        # one pass over data
        for i in range((data['size']+1)//params.batch_size):
            # fetch sentences and tags
            batch_sentences = [data['data'][idx] for idx in order[i*params.batch_size:(i+1)*params.batch_size]]
            batch_tags = [data['labels'][idx] for idx in order[i*params.batch_size:(i+1)*params.batch_size]]
            
            # compute length of longest sentence in batch
            seql = [len(s) for s in batch_sentences]
            batch_max_len = max(seql)

            # prepare a numpy array with the data, initialising the data with pad_ind and all labels with -1
            # initialising labels to -1 differentiates tokens with tags from PADding tokens
            batch_data = self.pad_ind*np.ones((len(batch_sentences), batch_max_len))
            batch_labels = -1*np.ones((len(batch_sentences), batch_max_len))

            # copy the data to the numpy array
            for j in range(len(batch_sentences)):
                cur_len = len(batch_sentences[j])
                batch_data[j][:cur_len] = batch_sentences[j]
                batch_labels[j][:cur_len] = batch_tags[j]

            # since all data are indices, we convert them to torch LongTensors
            batch_data, batch_labels = torch.LongTensor(batch_data), torch.LongTensor(batch_labels)

            # shift tensors to GPU if available
            if params.cuda:
                batch_data, batch_labels = batch_data.cuda(), batch_labels.cuda()

            # convert them to Variables to record operations in the computational graph
            batch_data, batch_labels = Variable(batch_data), Variable(batch_labels)
    
            yield batch_data, batch_labels, seql

