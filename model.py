import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init


class Encoder(nn.Module):
    '''
    A class representing the Encoder of the Seq2Seq model, responsible for
    encoding an input sentence into a semantic representation. 
    '''

    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=1, cuda='cpu'):
        '''
        Arguments:
            vocab_size: the vocabulary size of the input sentence language
            embedding_dim: the dimensionality of the embedding vectors
            hidden_size: the number of hidden neurons in the GRU
            num_layers: the number of layers in the GRU
            cuda: indication on whether to use CPU or GPU
        '''
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim).to(cuda)

        # Initialise weights of the embedding layer using a normal
        init.normal_(self.embedding.weight, 0.0, 0.2)

        # GRU layer
        self.gru = nn.GRU(
            embedding_dim,
            hidden_size,
            num_layers,
            batch_first=True
        ).to(cuda)

    def forward(self, word_inputs, hidden=None):
        '''
        Performs a forward pass of the GRU and returns
        the output at every time step, as well as the
        hidden state of the last timestep.

        Arguments:
            word_inputs: a matrix containing rows of index sentences.

        Returns:
            output: the output at each time step of the GRU
            hidden: the hidden state of the last time step of the GRU
        '''
        embedded = self.embedding(word_inputs)
        embedded = nn.ReLU()(embedded)

        if hidden is None:
            output, hidden = self.gru(embedded)

        else:
            output, hidden = self.gru(embedded, hidden)

        return output, hidden


class AttentionDecoder(nn.Module):
    '''
    A class that representation the decoder of the Seq2Seq with an
    attention mechanism, responsible for decoding the encoded 
    representation into a translated language.
    '''

    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=1, cuda='cpu'):
        '''
        Arguments:
            vocab_size: the vocabulary size of the translated language
            embedding_dim: the dimensionality of the embedding vectors
            hidden_size: the number of hidden neurons in the GRU + attention layers
            num_layers: the number of layers in the GRU layer
            cuda: indication on whether to use CPU or GPU
        '''
        super().__init__()

        self.cuda = cuda

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Embedding layer
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim=embedding_dim
        ).to(cuda)

        # Initialise weights of embedding layer using normal
        init.normal_(self.embedding.weight, 0.0, 0.2)

        # Attention layer, mapping the concatenation of the previous hidden
        # state of the decoder with an output at time t of the encoder into
        # a score (attention weight)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        ).to(cuda)

        # A layer that combines embedding with context from attention
        self.attention_combine = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU()
        ).to(cuda)

        # GRU layer
        self.gru = nn.GRU(
            embedding_dim,
            hidden_size,
            num_layers,
            batch_first=True
        ).to(cuda)

        # Final fully-connected layer mapping at each time step
        # to a vector the size of the vocabulary
        self.fc = nn.Linear(hidden_size, vocab_size).to(cuda)

        # Instantiate weights of the fully-connected layer with a normal
        init.normal_(self.fc.weight, 0.0, 0.2)

    def forward(self, word_inputs, hidden, output_encoder, retrieve_attention=False):
        '''
        Performs a forward pass of the decoder + attention to provide a translation
        at a time step t.

        Arguments:
            word_inputs: a matrix containing rows of index sentences
            hidden: the hidden state to pass into the GRU
            output_encoder: the hidden states at every time step of the encoder GPU.
            retrieve_attention (bool): returns attention weights if true

        Returns:
            output: the output of the decoder at time t
            hidden: the hidden state of the decoder at time t
            a_softmax_temp: attention weights at time t
        '''
        batch_size = output_encoder.shape[0]
        input_sentence_length = output_encoder.shape[1]

        # Attention coefficients
        a = torch.zeros((batch_size, input_sentence_length)).to(self.cuda)

        # For each element in encoder sequence
        for j in range(input_sentence_length):

            # Get the hidden state of the jth element in the encoder
            e_t = output_encoder[:, j]

            # Get the hidden state of the previous element in the decoder
            h_prev = hidden[0]

            # Concat these two hidden states
            e_h_concat = torch.cat((e_t, h_prev), dim=1)

            # Compute score between these two hidden states# 128 x 1
            a_t = self.attention(e_h_concat)
            a[:, j] = a_t[:, 0]

        # Softmax across time steps
        a_softmax = F.softmax(a, dim=1)
        a_softmax_temp = a_softmax

        a_softmax = a_softmax.view(batch_size, -1, 1)
        a_softmax = a_softmax.repeat(1, 1, self.hidden_size)

        # Combine encoder hidden states and attention to create context
        context = torch.sum(torch.mul(a_softmax, output_encoder),
                            dim=1).reshape(batch_size, 1, -1)

        # Convert word into embedding
        embedded = self.embedding(word_inputs).view(
            word_inputs.shape[0], 1, -1)

        # Concatenate the embedding and the context
        embedded_context_concat = torch.cat((embedded, context), dim=2)

        # Combine embedding and context via neural network
        attn_combined = self.attention_combine(embedded_context_concat)

        # Pass through the decoder
        output, hidden = self.gru(attn_combined, hidden)

        # Pass through final layer to map to vocab
        output = output.view(batch_size, -1)
        output = nn.ReLU()(output)
        output = self.fc(output)

        if retrieve_attention:
            return output, hidden, a_softmax_temp

        return output, hidden


class Seq2Seq(nn.Module):
    '''
    A class that represents the Seq2Seq model, combining the 
    encoder and decoder components.
    '''

    def __init__(self, input_vocab_size, output_vocab_size, embedding_dim, hidden_size, num_layers, cuda='cpu'):
        '''
        Arguments:
            input_vocab_size: the vocabulary size of the input language
            output_vocab_size: the vocabulary size of the output language
            embedding_dim: the dimensionality of the embedding vectors in both input and output language
            hidden_size: the number of hidden units to use in both encoder + decoder
            num_layers: the number of layers to use in both encoder + decoder
            cuda: indication on whether to use CPU or GPU
        '''
        super().__init__()
        self.cuda = cuda

        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Instantiate encoder
        self.encoder = Encoder(
            input_vocab_size, embedding_dim, hidden_size, num_layers, cuda=cuda).to(cuda)

        # Instantiate decoder + attention
        self.decoder = AttentionDecoder(
            output_vocab_size, embedding_dim, hidden_size, num_layers, cuda=cuda).to(cuda)

        # Instantiate dummy bleu1 metric
        self.bleu1 = -1

    def forward(self, x, y):
        '''
        Performs a forward pass of the Seq2Seq model

        Arguments:
            x: the indexed input sentences
            y: the true indexed output sentences

        Returns:
            output_sequence: a matrix containing the output sequence
                             predictions 
        '''
        # Encoder sentence into representation
        output_encoder, hidden = self.encoder(x)

        output_sentence_length = y.shape[1]
        batch_size = output_encoder.shape[0]

        # Store output sequence at every time step
        output_sequence = torch.zeros(
            (batch_size, output_sentence_length - 1, self.output_vocab_size))

        # True translations, ignoring the last <eos> element
        teacher_forcing_sequence = y[:, :-1]

        # For each step in the true output sequence
        for i in range(teacher_forcing_sequence.shape[1]):

            # Decode the representation
            output, hidden = self.decoder(
                teacher_forcing_sequence[:, i], hidden, output_encoder)

            # Store the output of each time step
            output_sequence[:, i] = output

        return output_sequence

    def get_attention_matrix(self, input_sentence_indices):
        '''
        Retrieves the attention matrix for a whole sentence

        Arguments:
            input_sentence_indices: a sentence in index form

        Returns:
            attention_matrix: the attention weight matrix
        '''
        output_encoder, hidden = self.encoder(input_sentence_indices)

        counter = 0
        current_pred = 0
        result = [current_pred]
        attention_matrix = []

        while current_pred != 1 and counter <= 10:
            current_output_word = torch.tensor([[current_pred]]).to(self.cuda)
            output, hidden, attention = self.decoder(
                current_output_word, hidden, output_encoder, retrieve_attention=True)

            attention_matrix.append(attention.data.cpu().numpy())

            softmax_output = F.softmax(output, dim=1)

            current_pred = torch.argmax(softmax_output, dim=1)

            result.append(current_pred.data.cpu().numpy()[0])
            counter += 1

        attention_matrix = np.array(
            attention_matrix).reshape(-1, input_sentence_indices.shape[1])

        return attention_matrix

    def predict(self, x):
        '''
        Given an input sentence, performs a translation prediction.

        Arguments:
            x: an input sentence in index form

        Returns:
            result: an output sentence in index form.
        '''
        output_encoder, hidden = self.encoder(x)

        counter = 0
        current_pred = 0
        result = [current_pred]

        while current_pred != 1 and counter <= 10:
            current_output_word = torch.tensor([[current_pred]]).to(self.cuda)
            output, hidden = self.decoder(
                current_output_word, hidden, output_encoder)

            softmax_output = F.softmax(output, dim=1)

            current_pred = torch.argmax(softmax_output, dim=1)

            result.append(current_pred.data.cpu().numpy()[0])
            counter += 1

        return result
