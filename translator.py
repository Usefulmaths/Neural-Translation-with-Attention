import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from model import Seq2Seq
from language import Preprocessor


class Translator(object):
    '''
    A class that represents the translator, which deals with training the model, 
    translating sentences, and calculating useful metrics to further understand
    the model.
    '''

    def __init__(self, english_language, norwegian_language, cuda='cpu'):
        '''
        Arguments:
            english language: a Language object containing information about English.
            norwegian_language: a Language object containing information about Norwegian.
            cuda: an indication on whether to use CPU or GPU.
        '''
        self.cuda = cuda

        # Instantiate Seq2Seq model
        self.model = self.instantiate_model(english_language.vocab_size(),
                                            norwegian_language.vocab_size())

        # Instantiate Preprocessor object
        self.preprocessor = Preprocessor()

        self.english_language = english_language
        self.norwegian_language = norwegian_language

    def instantiate_model(self, english_vocab_size, norwegian_vocab_size, embedding_dim=256, hidden_size=256, num_layers=1):
        '''
        Instantiates the Seq2Seq model. 

        Arguments:
            english_vocab_size: the vocabulary size of English
            norwegian_vocab_size: the vocabulary size of Norwegian
            embedding_dim: the dimensionality of the embedding layers
            hidden_size: the number of neurons in each hidden layer
            num_layers: the number of hidden layers

        Returns:
            seq2seq_model: the instantiated Seq2Seq model.
        '''
        seq2seq_model = Seq2Seq(english_vocab_size,
                                norwegian_vocab_size,
                                embedding_dim,
                                hidden_size,
                                num_layers,
                                cuda=self.cuda).to(self.cuda)

        return seq2seq_model

    def train(self, X_train, y_train, X_test=None, y_test=None, epochs=500, train_batch_size=128, test_batch_size=128, log_batch=20, early_stopping_criteria=50, file_save_name=None, file_load_name=None):
        '''
        Performs training on the Seq2Seq model. 

        Arguments:
            X_train: the input training sentence data
            y_train: the output training sentence data
            X_test: the input testing sentence data
            y_test: the output testing sentence data
            epoch: the number of epochs to train for
            train_batch_size: batch size for training
            test_batch_size: batch size for testing
            log_batch: the number of batches to process before logging progress
            early_stopping_criteria: how many batches to process without 
                                     improvement in the metric before stopping
            file_save_name: file name to save model
            file_load_name: file name to load model
        '''

        # If a file_load_name is specified, load model
        if file_load_name is not None:
            self.load_model(file_load_name)

        # Convert training data into PyTorch tensors
        X_train = torch.tensor(X_train).to(self.cuda)
        y_train = torch.tensor(y_train).to(self.cuda)

        # Creates a DataSet + DataLoader for training data
        train_dataset = SentenceDataSet(X_train, y_train)
        train_dataloader = DataLoader(
            train_dataset, batch_size=train_batch_size, shuffle=True)

        # If test data given, create a Test DataSet + DataLoader
        if X_test is not None:
            X_test = torch.tensor(X_test).to(self.cuda)
            y_test = torch.tensor(y_test).to(self.cuda)

            test_dataset = SentenceDataSet(X_test, y_test)
            test_dataloader = DataLoader(
                test_dataset, batch_size=test_batch_size, shuffle=True)

        # Instantiate Adam Optimiser
        optim = torch.optim.Adam(self.model.parameters(), lr=0.001)

        # Instantiate CrossEntropyLoss
        cross_entropy = nn.CrossEntropyLoss()

        # Early stopping counter
        early_stopping = 0

        # For each epoch
        for epoch in range(epochs):
            # If the early stopping counter has reached the criteria
            # stop training
            if early_stopping >= early_stopping_criteria:
                break

            # For each batch in train DataLoader
            for batch_idx, (x, y) in enumerate(train_dataloader):
                # If the early stopping counter has reached the criteria
                # stop training
                if early_stopping >= early_stopping_criteria:
                    break

                # Send batch input and output to GPU
                x = x.to(self.cuda)
                y = y.to(self.cuda)

                # Forward pass of Seq2Seq model
                output_sentence = self.model(x, y).to(
                    self.cuda)
                target_sentence = y[:, 1:]

                # Store loss of each example in the batch
                total_loss = torch.zeros(
                    (train_batch_size, output_sentence.shape[1]))

                # For each time step, calculate the Cross entropy
                # between the predicted output and the target output.
                for i in range(output_sentence.shape[1]):
                    loss = cross_entropy(
                        output_sentence[:, i], target_sentence[:, i])

                    total_loss[:, i] = loss

                # Take the mean of the loss of each time step
                total_loss = total_loss.mean()

                # Zero grad the optimiser
                optim.zero_grad()

                # Perform backpropagation to retrieve
                # gradients of loss with respect to all the
                # parameters in the network
                total_loss.backward()

                # Clip gradients
                torch.nn.utils.clip_grad_norm(seq2seq_model.parameters(), 3.0)

                # Performs a gradient step
                optim.step()

                # If batch is a multiple of log_batch
                # show progress of model on test
                if batch_idx % log_batch == 0:

                    # For each batch in test DataLoader
                    for batch_idy, (x, y) in enumerate(test_dataloader):

                        # Send input and output batch to GPU
                        x = x.to(self.cuda)
                        y = y.to(self.cuda)

                        # Perform forward pass of Seq2Seq
                        output_sentence = self.model(x, y).to(self.cuda)
                        target_sentence = y[:, 1:]

                        # Store loss for each time step of the decoder
                        total_test_loss = torch.zeros(
                            (test_batch_size, output_sentence.shape[1]))

                        # If batch the size of test_batch_size
                        if x.shape[0] == test_batch_size:

                            # For each time step, calculate the Cross entropy
                            # between the predicted output and the target output.
                            for i in range(output_sentence.shape[1]):
                                loss = cross_entropy(
                                    output_sentence[:, i], target_sentence[:, i])
                                total_test_loss[:, i] = loss

                            # Calculate mean loss over time steps
                            total_test_loss = total_test_loss.mean()

                            # Argmax the output_sentence to retrieve prediction indices.
                            preds = torch.argmax(output_sentence, dim=2)

                            # Calculate the bleu scores (1, 2, 3, 4)
                            bleu_scores = self.bleu_score_batch(
                                target_sentence, preds)

                            bleu1 = bleu_scores[0]

                            # If model instantiates, override the instance
                            if self.model.bleu1 == -1:
                                self.model.bleu1 = bleu1

                            # If performance hasn't improved, add
                            # to the early stopping counter
                            if bleu1 < self.model.bleu1:
                                early_stopping += 1

                            # Else, update the best bleu1 and reset
                            # the early stopping counter
                            else:
                                self.model.bleu1 = bleu1
                                early_stopping = 0

                                # If file save name specified,
                                # save the model
                                if file_save_name is not None:
                                    torch.save(self.model.state_dict(), '%s' %
                                               file_save_name)

                        # Perform for only one batch
                        break

                    # Print the progress
                    print("Epoch %d, Batch %d -> Train Loss: %.4f\tTest Loss: %.4f\tBleu-1: %.4f (ES: %d)\tBleu-2: %.4f\tBleu-3: %.4f\tBleu-4: %.4f" %
                          (epoch, batch_idx, total_loss, total_test_loss, bleu1, early_stopping, bleu_scores[1], bleu_scores[2], bleu_scores[3]))

    def load_model(self, file_name):
        '''
        Loads an already trained model.

        Arguments:
            file_name: the name of the file that contains the model to load
        '''

        # Load onto the CPU
        if self.cuda == 'cpu':
            self.model.load_state_dict(
                torch.load(file_name, map_location='cpu'))

        # Load onto the GPU
        else:
            self.model.load_state_dict(torch.load(file_name))

    def translate(self, sentence):
        '''
        Given a sentence in the input language, translates 
        into the output language

        Arguments:
            sentence: a sentence in the input language

        Returns:
            translation: the translated sentence in the output language.
        '''
        # Preprocess
        preprocessor = Preprocessor()
        sentence = preprocessor.preprocess_sentence(sentence)

        # Convert to indices
        indices = self.english_language.sentence_to_idx(sentence)

        indices = torch.tensor(
            indices).view(1, len(indices))

        # Norwegian indices prediction
        prediction = self.model.predict(indices.to(self.cuda))

        # Convert Norwegian indices to sentence
        translation = self.norwegian_language.idx_to_sentence(prediction)

        # Remove <sos> and <eos>
        translation = [word for word in translation if word != '<sos>']
        translation = [word for word in translation if word != '<eos>']

        translation = ' '.join(translation)

        return translation

    def get_attention_matrix(self, sentence):
        '''
        Returns the attention matrix of a sentence as passed through the model.

        Arguments:
            sentence: a sentence in the input language

        Returns:
            attention_matrix: the attention weight matrix
        '''
        # Preprocess
        preprocessor = Preprocessor()
        sentence = preprocessor.preprocess_sentence(sentence)

        # Convert to indices
        indices = self.english_language.sentence_to_idx(sentence)

        indices = torch.tensor(
            indices).view(1, len(indices))

        # Retrieve attention matrix from model.
        attention_matrix = self.model.get_attention_matrix(
            indices.to(self.cuda))

        return attention_matrix

    def bleu_score_batch(self, target_sents, predicted_sents):
        '''
        Calcualtes the BLEU (1, 2, 3, 4) score for a batch of sentences

        Arguments:
            target_sent: a list of target sentences
            predicted_sent: a list of predicted sentences

        Returns:
            avg_bleus: the average BLEU (1, 2, 3, 4) score over the batch
        '''
        batch_size = len(target_sents)

        bleus = np.zeros((batch_size, 4))

        for i in range(len(target_sents)):
            bleu1, bleu2, bleu3, bleu4 = self.bleu_score(
                target_sents[i], predicted_sents[i])
            bleus[i][0] = bleu1
            bleus[i][1] = bleu2
            bleus[i][2] = bleu3
            bleus[i][3] = bleu4

        avg_bleus = np.mean(bleus, axis=0)

        return avg_bleus

    def bleu_score(self, target_sent, predicted_sent):
        '''
        Given a single target sentence and a predicted sentence, 
        calculates the BLEU (1, 2, 3, 4) score.

        Arguments:
            target_sent: the target sentence
            predicted_sent: the predicted sentence

        Returns:
            bleu1: the BLEU-1 score
            bleu2: the BLEU-2 score
            bleu3: the BLEU-3 score
            bleu4: the BLEU-4 score
        '''
        if type(target_sent) is torch.Tensor:
            target_sent = target_sent.data.cpu().numpy()
            predicted_sent = predicted_sent.data.cpu().numpy()

            # Remove sos(0), eos(1), and padding(3)
            target_sent = [idx for idx in target_sent if idx !=
                           0 and idx != 1 and idx != 3]
            predicted_sent = [
                idx for idx in predicted_sent if idx != 0 and idx != 1 and idx != 3]

        else:
            target_sent = [word for word in target_sent if word !=
                           '<sos>' and word != '<eos>' and word != '<pad>']
            predicted_sent = [
                word for word in predicted_sent if word != '<sos>' and word != '<eos>' and word != '<pad>']

        # If predicted sentence empty
        if len(predicted_sent) == 0:
            return 0, 0, 0, 0

        # Smoothing to deal with short sentences
        smoothie = SmoothingFunction().method1

        bleu1 = sentence_bleu(
            [target_sent], predicted_sent, weights=(1, 0, 0, 0), smoothing_function=smoothie)
        bleu2 = sentence_bleu(
            [target_sent], predicted_sent, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
        bleu3 = sentence_bleu([target_sent], predicted_sent,
                              weights=(1./3, 1./3, 1./3, 0), smoothing_function=smoothie)
        bleu4 = sentence_bleu([target_sent], predicted_sent,
                              weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)

        return bleu1, bleu2, bleu3, bleu4


class SentenceDataSet(Dataset):
    '''
    A class that acts as a Dataset object for a pair of 
    English and Norwegian sentences. Allows for easy batching.
    '''

    def __init__(self, english_sentences, norwegian_sentences):
        self.english_sentences = english_sentences
        self.norwegian_sentences = norwegian_sentences

    def __len__(self):
        return len(self.english_sentences)

    def __getitem__(self, index):
        return (self.english_sentences[index], self.norwegian_sentences[index])
