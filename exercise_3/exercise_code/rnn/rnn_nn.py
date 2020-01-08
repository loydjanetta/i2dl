import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=20, activation="tanh"):
        super().__init__()
        """
        Inputs:
        - input_size: Number of features in input vector
        - hidden_size: Dimension of hidden vector
        - activation: Nonlinearity in cell; 'tanh' or 'relu'
        """
        #######################################################################
        # TODO: Build a simple one layer RNN with an activation with the      #
        # attributes defined above and a forward function below. Use the      #
        # nn.Linear() function as your linear layers.                         #
        # Initialse h as 0 if these values are not given.                     #
        #######################################################################
        self.hidden_size = hidden_size
		
        assert activation in ["tanh", "relu"]

        if activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU(inplace=True)

        self.linear_x = nn.Linear(input_size, hidden_size)
        self.linear_h = nn.Linear(hidden_size, hidden_size)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x, h=None):
        """
        Inputs:
        - x: Input tensor (seq_len, batch_size, input_size)
        - h: Optional hidden vector (nr_layers, batch_size, hidden_size)

        Outputs:
        - h_seq: Hidden vector along sequence
                 (seq_len, batch_size, hidden_size)
        - h: Final hidden vetor of sequence(1, batch_size, hidden_size)
        """
        h_seq = []
        #######################################################################
        #                                YOUR CODE                            #
        #######################################################################
        seq_len, batch_size, input_size = x.shape

        if h is None:
            h = torch.zeros((1, batch_size, self.hidden_size))

        x_ = self.linear_x(x[0])
        h = self.linear_h(h)

        h_seq.append(self.activation(h + x_))

        for t in range(1, seq_len):
            temp = self.linear_h(h_seq[t-1]) + self.linear_x(x[t])
            h_seq.append(self.activation(temp))

        h = h_seq[-1]
        h_seq = torch.cat(h_seq, dim=0)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
        return h_seq, h


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=20):
        super().__init__()
        #######################################################################
        # TODO: Build a one layer LSTM with an activation with the attributes #
        # defined above and a forward function below. Use the                 #
        # nn.Linear() function as your linear layers.                         #
        # Initialse h and c as 0 if these values are not given.               #
        #######################################################################
        self.hidden_size = hidden_size

        self.linear_f_x = nn.Linear(input_size, hidden_size)
        self.linear_f_h = nn.Linear(hidden_size, hidden_size)

        self.linear_i_x = nn.Linear(input_size, hidden_size)
        self.linear_i_h = nn.Linear(hidden_size, hidden_size)

        self.linear_o_x = nn.Linear(input_size, hidden_size)
        self.linear_o_h = nn.Linear(hidden_size, hidden_size)

        self.linear_c_x = nn.Linear(input_size, hidden_size)
        self.linear_c_h = nn.Linear(hidden_size, hidden_size)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x, h=None, c=None):
        """
        Inputs:
        - x: Input tensor (seq_len, batch_size, input_size)
        - h: Hidden vector (nr_layers, batch_size, hidden_size)
        - c: Cell state vector (nr_layers, batch_size, hidden_size)

        Outputs:
        - h_seq: Hidden vector along sequence
                 (seq_len, batch_size, hidden_size)
        - h: Final hidden vector of sequence(1, batch_size, hidden_size)
        - c: Final cell state vector of sequence(1, batch_size, hidden_size)
        """
        h_seq = None
        #######################################################################
        #                                YOUR CODE                            #
        #######################################################################
        h_seq = []
        c_seq = []
        seq_len, batch_size, input_size = x.shape

        if h is None:
            h = torch.zeros((1, batch_size, self.hidden_size))
        if c is None:
            c = torch.zeros((1, batch_size, self.hidden_size))

        f_x = self.linear_f_x(x[0])
        f_h = self.linear_f_h(h)

        i_x = self.linear_i_x(x[0])
        i_h = self.linear_i_h(h)

        o_x = self.linear_o_x(x[0])
        o_h = self.linear_o_h(h)

        c_x = self.linear_c_x(x[0])
        c_h = self.linear_c_h(h)

        f = self.sigmoid(f_x + f_h)
        i = self.sigmoid(i_x + i_h)
        o = self.sigmoid(o_x + o_h)

        c_seq.append(f * c + i * self.tanh(c_x + c_h))
        h_seq.append(o * self.tanh(c_seq[-1]))

        for t in range(1, seq_len):
            f_t = self.sigmoid(self.linear_f_x(x[t]) + self.linear_f_h(h_seq[-1]))

            i_t = self.sigmoid(self.linear_i_x(x[t]) + self.linear_i_h(h_seq[-1]))

            o_t = self.sigmoid(self.linear_o_x(x[t]) + self.linear_o_h(h_seq[-1]))

            # c_seq.append(f_t * c_seq[-1] + i_t * self.tanh(c_x[t] + c_h))
            c_seq.append(f_t * c_seq[-1] + \
                         i_t * self.tanh(self.linear_c_x(x[t]) + \
                         self.linear_c_h(h_seq[-1]))
                         )
            h_seq.append(o_t * self.tanh(c_seq[-1]))

        h = h_seq[-1]
        c = c_seq[-1]
        h_seq = torch.cat(h_seq, dim=0)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
        return h_seq, (h, c)


class RNN_Classifier(torch.nn.Module):
    def __init__(self, classes=10, input_size=28, hidden_size=128,
                 activation="relu"):
        super(RNN_Classifier, self).__init__()
        #######################################################################
        #  TODO: Build a RNN classifier                                       #
        #######################################################################
        self.rnn = nn.RNN(input_size, hidden_size, nonlinearity=activation)
        self.fc = nn.Linear(hidden_size, classes)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        x = self.rnn(x)[1]
        x = self.fc(x)[0]
        return x

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)


class LSTM_Classifier(torch.nn.Module):
    def __init__(self, classes=10, input_size=28, hidden_size=128):
        super(LSTM_Classifier, self).__init__()
        #######################################################################
        #  TODO: Build a LSTM classifier                                      #
        #######################################################################
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, classes)

        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################

    def forward(self, x):
        x = self.lstm(x)[1][0]
        x = self.fc(x)[0]
        return x

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

