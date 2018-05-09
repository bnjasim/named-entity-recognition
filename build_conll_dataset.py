"""Read, split and save the kaggle dataset for our model"""

import os
import sys


def load_dataset(path_dataset):
    """Loads dataset into memory from file"""
    with open(path_dataset, encoding='ISO-8859-1') as f:
        dataset = []
        words, tags = [], []

        for line in f:
            line = line.strip()
            if (len(line) == 0 or line.startswith("-DOCSTART-")):
                if len(words) > 0:
                    assert len(words) == len(tags)
                    dataset.append((words, tags))
                    words, tags = [], []
            else:
                try:
                    ls = line.split(' ')
                    word, tag = ls[0],ls[-1]
                    word, tag = str(word), str(tag)
                    words.append(word)
                    tags.append(tag)
                except UnicodeDecodeError as e:
                    print("An exception was raised, skipping a word: {}".format(e))

    return dataset


def save_dataset(dataset, save_dir):
    """Writes sentences.txt and labels.txt files in save_dir from dataset

    Args:
        dataset: ([(["a", "cat"], ["O", "O"]), ...])
        save_dir: (string)
    """
    # Create directory if it doesn't exist
    print("Saving in {}...".format(save_dir))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Export the dataset
    with open(os.path.join(save_dir, 'sentences.txt'), 'w') as file_sentences:
        with open(os.path.join(save_dir, 'labels.txt'), 'w') as file_labels:
            for words, tags in dataset:
                file_sentences.write("{}\n".format(" ".join(words)))
                file_labels.write("{}\n".format(" ".join(tags)))
    print("- done.")


if __name__ == "__main__":
    # Check that the dataset exists (you need to make sure you haven't downloaded the `ner.csv`)
    path_train = 'data/coNLL/eng/eng.train'
    path_valid = 'data/coNLL/eng/eng.testa'
    path_test = 'data/coNLL/eng/eng.testb'

    msg = "{} file not found. Make sure you have downloaded the right dataset".format(path_train)
    assert os.path.isfile(path_train), msg

    # Load the dataset into memory
    print("Loading CoNLL dataset into memory...")
    train_dataset = load_dataset(path_train)
    valid_dataset = load_dataset(path_valid)
    test_dataset = load_dataset(path_test)
    print("- done.")

    # Save the datasets to files
    save_dataset(train_dataset, 'data/coNLL/eng/train')
    save_dataset(valid_dataset, 'data/coNLL/eng/val')
    save_dataset(test_dataset, 'data/coNLL/eng/test')
