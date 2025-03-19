"""
This module contains classes to build Seq2Seq models, including trace-based 
Seq2Seq model (class: Seq2Seq_trace) and integrated Seq2Seq model (class:
Seq2Seq_cat).

Classes:

    Encoder
    Decoder
    Seq2Seq_trace
    Seq2Seq_cat
    
"""
import torch
import torch.nn as nn
import random
import torch.nn.functional as F

class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, dropout, bidirect):
        """

        Parameters
        ----------
        input_size: int
            Number of input features
        hidden_size: int
            Number of hidden units in LSTM
        num_layers: int
            Number of stacked LSTM layers
        dropout: float
            Dropout probability to drop out the outputs of each LSTM layer 
            except the last layer.
        bidirect: boolean
            Indicating whether the model is birectional or not.

        """

        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, 
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=True,
                           dropout=dropout,
                           bidirectional=bidirect)
        self.bidirect = bidirect
        
    def forward(self, prefix):
        """

        Parameters
        ----------
        prefix: tensor
            Trace prefix or log prefix
            shape: (batch_size, prefix_len, num_features)

        Returns
        -------
        hidden: tensor
            shape: (num_layers(*2 if bidirectional), batch_size, hidden_size)
        cell: tensor
            shape: (num_layers(*2 if bidirectional), batch_size, hidden_size)

        """
        
        outputs, (hidden, cell) = self.lstm(prefix)
        # outputs shape: (batch_size, prefix_len, hidden_size (*2 if bidirectional))
        # hidden shape: (num_layers(*2 if bidirectional), batch_size, hidden_size)
        # cell shape: (num_layers(*2 if bidirectional), batch_size, hidden_size)

        if self.bidirect:
            hidden_1 = hidden[0::2,:,:]
            hidden_2 = hidden[1::2,:,:]
            hidden = torch.cat((hidden_1, hidden_2), -1)
            # hidden shape: (num_layers, batch_size, hidden_size * 2)

            cell_1 = cell[0::2,:,:]
            cell_2 = cell[1::2,:,:]
            cell = torch.cat((cell_1, cell_2), -1)
            # cell shape: (num_layers, batch_size, hidden_size * 2)

        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        """

        Parameters
        ----------
        input_size: int
            Number of input features
        hidden_size: int
            Number of hidden units in LSTM.
        output_size: int
            Number of features for output.
        num_layers: int
            Number of stacked LSTM layers
        dropout: float
            Dropout probability to drop out the outputs of each LSTM layer 
            except the last layer.

        """        
        super(Decoder, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size , 
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=True,
                           dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, suffix_vector, hidden, cell):

        """
        Parameters
        ----------
        suffix_vector: tensor
            Feature vector representing an event in the suffix
            shape: (batch_size, num_features)
        hidden: tensor
            Hidden state
            shape: (num_layers, batch_size, hidden_size)
        cell: tensor
            Cell state
            shape: (num_layers, batch_size, hidden_size)

        Returns
        -------
        prediction: tensor
            shape: (batch_size, output_size)
        hidden: tensor
            shape: (num_layers, batch_size, hidden_size)
        cell: tensor
            shape: (num_layers, batch_size, hidden_size)
        """
         
        suffix_vector = suffix_vector.unsqueeze(1) 
        # suffix_vector shape: (batch_size, 1, num_features) 
        
        # nn.LSTM requires arguments: input, (h_0, c_0)
        outputs, (hidden, cell) = self.lstm(suffix_vector, (hidden, cell)) 
        # outputs shape: (batch_size, 1, hidden_size)
        
        # nn.Linear: input shape: (*, input_size); output shape: (*, output_size),
        # where * means any number of dimensions including none
        prediction = self.fc(outputs) 
        # prediction shape: (batch_size, 1, output_size)
        
        # in Seq2Seq model, prediction will be stored using: 
        # predictions[:, t, :] = prediction
        # which requires prediction to be a tensor of shape (batch_size, output_size)
        prediction = prediction.squeeze(1) 
        # prediction shape: (batch_size, output_size)

        return prediction, hidden, cell

class Seq2Seq_trace(nn.Module): 
    """
    One encoder, two decoders

    """            
    def __init__(self, num_act, encoder, act_decoder, time_decoder):
        """
        Parameters
        ----------
        num_act: int
            Number of activity labels (including padding, SOC, EOC, unknown label)
        encoder: object
            An instance of Class Encoder used to encode log/trace prefix.
        act_dncoder: object
            An instance of Class Decoder used to generate activity label suffix prediction.
        time_dncoder: object
            An instance of Class Decoder used to generate timestamp suffix prediction.
        """
        super(Seq2Seq_trace, self).__init__()
        self.num_act = num_act
        self.encoder = encoder
        self.act_decoder = act_decoder
        self.time_decoder = time_decoder

    def forward(self, log_prefix_tensor, trace_prefix_tensor, trace_act_suffix_tensor, 
                trace_time_suffix_tensor, teacher_force_ratio):
        """
        Parameters
        ----------
        log_prefix_tensor: tensor
            shape: (batch_size, prefix_len, num_act + 1),
            actually will not be used by the function. keep this parameter to 
            make training function more general
        trace_prefix_tensor: tensor
            shape: (batch_size, prefix_len, num_act + 2),
            input of Seq2Seq model
        trace_act_suffix_tensor: tensor
            shape: (batch_size, suffix_len),
            target of Seq2Seq model
        trace_time_suffix_tensor: tensor
            shape: (batch_size, suffix_len),
            target of Seq2Seq model
        teacher_force_ratio: float
            The probability that the ground truth will be used for prediction 
            generation  
        
        Returns
        -------
        act_predictions: tensor
            shape: (batch_size, suffix_len, num_act)
        time_predictions: tensor
            shape: (batch_size, suffix_len, 1)
        """
        
        batch_size = trace_prefix_tensor.shape[0]

        # the last hidden state of the encoder is used as the initial hidden 
        # state of both decoders
        hidden, cell = self.encoder(trace_prefix_tensor)
        # hidden shape: (num_layers, batch_size, hidden_size)
        # cell shape: (num_layers, batch_size, hidden_size)

        # -- for activity decoder --
        
        # one-hot encode the ground truth trace_act_suffix_tensor to use as 
        # input during teacher forcing
        trace_act_suffix_tensor = F.one_hot(trace_act_suffix_tensor, num_classes=self.num_act)
        trace_act_suffix_tensor[:, :, 0] = 0 # Ensure padding is represented by all 0s
        # trace_act_suffix_tensor shape: (batch_size, suffix_len, num_act)

        # initialize the tensor that store decoder outputs
        act_suffix_len = trace_act_suffix_tensor.shape[1]
        act_predictions = torch.zeros(batch_size, act_suffix_len, self.num_act)

        # grab the activity label (one-hot encoded) of last event in prefix as 
        # the input for the first state of activity decoder
        # for prefix tensor, the last dimension is num_act + 1 (log) or 2 (trace)
        x_act = trace_prefix_tensor[:, -1, :self.num_act]
        # x_act shape: (batch_size, num_act)

        # store (hidden, cell) in new variables, otherwise timestamp decoder 
        # will use (hidden, cell) already changed by activity decoder
        act_hidden, act_cell = hidden, cell

        # generate prediction step by step
        for t in range(act_suffix_len):
            # use hidden, cell from encoder as context for the first state in the 
            # decoder, and in later states, outputs, hidden, cell from previous 
            # state are used as inputs for decoder
            act_prediction, act_hidden, act_cell = self.act_decoder(x_act, act_hidden, act_cell)
            # act_prediction shape: (batch_size, num_act)

            # store prediction
            act_predictions[:, t, :] = act_prediction

            # get the best actibity label (index) the decoder predicts
            best_guess = act_prediction.argmax(1)
            # best_guess shape: (batch_size)
            best_guess = F.one_hot(best_guess, num_classes=self.num_act)
            # best_guess shape: (batch_size, num_act)
            best_guess[:, 0] = 0 # Ensure padding is represented by all 0s

            # teacher forcing
            x_act = trace_act_suffix_tensor[:, t, :] if random.random() < teacher_force_ratio else best_guess
            # trace_act_suffix_tensor is long tensor. nn.LSTM expects input to 
            # be float tensor
            x_act = x_act.float()
            # x_act shape: (batch_size, num_act)

        # -- for timestamp decoder --

        trace_time_suffix_tensor = trace_time_suffix_tensor.unsqueeze(-1)
        # trace_time_suffix_tensor shape: (batch_size, suffix_length, 1)

        # initialize predictions tensor
        time_suffix_len = trace_time_suffix_tensor.shape[1]
        time_predictions = torch.zeros(batch_size, time_suffix_len, 1)

        # grab the timestamp of the last event in prefix as the first input for 
        # timestamp decoder
        # for model_trace, grab trace_ts_pre, the same as the target
        # for model_log, grab log_ts_pre, different from the target
        x_time = trace_prefix_tensor[:, -1, -1]
        # x_time shape: (batch_size)
        # reshape x_time to fit the required input shape of time_decoder
        x_time = x_time.unsqueeze(-1)
        # x_time shape: (batch_size, 1)

        # store (hidden, cell) in new variables
        time_hidden, time_cell = hidden, cell

        # generate prediction step by step
        for t in range(time_suffix_len):
            # use hidden, cell from encoder as context for the first state in the 
            # decoder, and in later states, outputs, hidden, cell from previous 
            # state are used as inputs for decoder
            time_prediction, time_hidden, time_cell = self.time_decoder(x_time, time_hidden, time_cell)
            # time_prediction shape: (batch_size, 1)

            # store prediction
            time_predictions[:, t, :] = time_prediction

            # teacher forcing
            x_time = trace_time_suffix_tensor[:, t, :] if random.random() < teacher_force_ratio else time_prediction
            # x_time shape: (batch_size, 1)

        return act_predictions, time_predictions
        
class Seq2Seq_cat(nn.Module): 
    """
    Two encoders, two decoders

    """           
    def __init__(self, num_act, log_encoder, trace_encoder, act_cat_decoder, time_cat_decoder):
        """
        Parameters
        ----------
        num_act: int
            Number of activity labels (including padding, SOC, EOC, unknown label)
        log_encoder: object
            An instance of Class Encoder used to encode log prefix.
        trace_encoder: object
            An instance of Class Encoder used to encode trace prefix.
        act_cat_dncoder: object
            An instance of Class Decoder used to generate activity label suffix prediction.
        time_cat_dncoder: object
            An instance of Class Decoder used to generate timestamp suffix prediction.
        """
        super(Seq2Seq_cat, self).__init__()
        self.num_act = num_act
        self.log_encoder = log_encoder
        self.trace_encoder = trace_encoder
        self.act_decoder = act_cat_decoder
        self.time_decoder = time_cat_decoder

    def forward(self, log_prefix_tensor, trace_prefix_tensor, trace_act_suffix_tensor, trace_time_suffix_tensor, teacher_force_ratio):
        """
        Parameters
        ----------
        log_prefix_tensor: tensor
            shape: (batch_size, prefix_len, num_act + 1),
            input of Seq2Seq model
        trace_prefix_tensor: tensor
            shape: (batch_size, prefix_len, num_act + 2),
            input of Seq2Seq model
        trace_act_suffix_tensor: tensor
            shape: (batch_size, suffix_len),
            target of Seq2Seq model
        trace_time_suffix_tensor: tensor
            shape: (batch_size, suffix_len),
            target of Seq2Seq model
        teacher_force_ratio: float
            The probability that the ground truth will be used for prediction 
            generation  
        
        Returns
        -------
        act_predictions: tensor
            shape: (batch_size, suffix_len, num_act)
        time_predictions: tensor
            shape: (batch_size, suffix_len, 1)
        """
        
        batch_size = log_prefix_tensor.shape[0]

        # --- The part below is different from Seq2Seq_one_input class --
        log_hidden, log_cell = self.log_encoder(log_prefix_tensor)
        # log_hidden shape: (num_layers, batch_size, enc_hidden_size)
        # log_cell shape: (num_layers, batch_size, enc_hidden_size)
        trace_hidden, trace_cell = self.trace_encoder(trace_prefix_tensor)
        # trace_hidden shape: (num_layers, batch_size, enc_hidden_size)
        # trace_cell shape: (num_layers, batch_size, enc_hidden_size)

        hidden = torch.cat((log_hidden, trace_hidden), -1)
        # hidden shape: (num_layers, batch_size, enc_hidden_size * 2)
        cell = torch.cat((log_cell, trace_cell), -1)
        # cell shape: (num_layers, batch_size, enc_hidden_size * 2)
        # --- The part above is different from Seq2Seq_one_input class --

        # -- for activity decoder --
        
        trace_act_suffix_tensor = F.one_hot(trace_act_suffix_tensor, num_classes=self.num_act)
        trace_act_suffix_tensor[:, :, 0] = 0
        
        act_suffix_len = trace_act_suffix_tensor.shape[1]
        act_predictions = torch.zeros(batch_size, act_suffix_len, self.num_act)

        x_act = trace_prefix_tensor[:, -1, :self.num_act]

        act_hidden, act_cell = hidden, cell

        for t in range(act_suffix_len):

            act_prediction, act_hidden, act_cell = self.act_decoder(x_act, act_hidden, act_cell)
            act_predictions[:, t, :] = act_prediction

            best_guess = act_prediction.argmax(1)
            best_guess = F.one_hot(best_guess, num_classes=self.num_act)
            best_guess[:, 0] = 0

            x_act = trace_act_suffix_tensor[:, t, :] if random.random() < teacher_force_ratio else best_guess
            x_act = x_act.float()

        # -- for timestamp decoder --

        trace_time_suffix_tensor = trace_time_suffix_tensor.unsqueeze(-1)

        time_suffix_len = trace_time_suffix_tensor.shape[1]
        time_predictions = torch.zeros(batch_size, time_suffix_len, 1)

        x_time = trace_prefix_tensor[:, -1, -1]
        x_time = x_time.unsqueeze(-1)

        time_hidden, time_cell = hidden, cell

        for t in range(time_suffix_len):

            time_prediction, time_hidden, time_cell = self.time_decoder(x_time, time_hidden, time_cell)
            time_predictions[:, t, :] = time_prediction

            x_time = trace_time_suffix_tensor[:, t, :] if random.random() < teacher_force_ratio else time_prediction

        return act_predictions, time_predictions