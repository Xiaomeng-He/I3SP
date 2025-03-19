import torch
import torch.nn as nn
import random
import numpy as np
import torch.nn.functional as F

class SEP_LSTM_trace(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, dropout, bidirect,
                 act_output_size, time_output_size):
        
        super(SEP_LSTM_trace, self).__init__()
        
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size,
                                  hidden_size=hidden_size,
                                  num_layers=num_layers,
                                  batch_first=True,
                                  dropout=dropout,
                                  bidirectional=bidirect)
        
        # Indicator of bidirectional layer
        if bidirect:
            bi = 2
        else:
            bi = 1

        self.act_fc = nn.Linear(hidden_size*bi, 
                                act_output_size)
        self.time_fc = nn.Linear(hidden_size*bi, 
                                 time_output_size)

    def forward(self, log_prefix, trace_prefix):
        """
        Parameters
        ----------
        log_prefix: tensor
            shape: (batch_size, trace_prefix_len, num_features)
        trace_prefix: tensor
            shape: (batch_size, log_prefix_len, num_features)

        Returns
        -------
        act_prediction: tensor
            shape: (batch_size, num_act)
        time_prediction: tensor
            shape: (batch_size, 1)
        """
        outputs, _ = self.lstm(trace_prefix)
        # outputs shape: (batch_size, prefix_len, hidden_size (*2 if bidirectional))
        outputs = outputs[:, -1, :]
        # outputs shape: (batch_size, hidden_size (*2 if bidirectional))

        # nn.Linear: input shape: (*, input_size); output shape: (*, output_size),
        # where * means any number of dimensions including none
        act_prediction = self.act_fc(outputs) 
        time_prediction = self.time_fc(outputs)
        # act_prediction shape: (batch_size, num_act)
        # time_prediction shape: (batch_size, 1)

        return act_prediction, time_prediction
            
class SEP_LSTM_cat(nn.Module):

    def __init__(self, trace_input_size, log_input_size, hidden_size, num_layers, 
                 dropout, bidirect, act_output_size, time_output_size):
        
        super(SEP_LSTM_cat, self).__init__()

        self.num_layers = num_layers

        self.trace_lstm = nn.LSTM(input_size=trace_input_size,
                                  hidden_size=hidden_size,
                                  num_layers=num_layers,
                                  batch_first=True,
                                  dropout=dropout,
                                  bidirectional=bidirect)
        
        self.log_lstm = nn.LSTM(input_size=log_input_size,
                                  hidden_size=hidden_size,
                                  num_layers=num_layers,
                                  batch_first=True,
                                  dropout=dropout,
                                  bidirectional=bidirect)
        
        # Indicator of bidirectional layer
        if bidirect:
            bi = 2
        else:
            bi = 1
        
        self.act_fc = nn.Linear(hidden_size*bi*2, 
                                act_output_size)

        self.time_fc = nn.Linear(hidden_size*bi*2, 
                                 time_output_size)

    def forward(self, log_prefix, trace_prefix):
        """
        Parameters
        ----------
        log_prefix: tensor
            shape: (batch_size, trace_prefix_len, num_features)
        trace_prefix: tensor
            shape: (batch_size, log_prefix_len, num_features)

        Returns
        -------
        hidden: tensor
            shape: (num_layers, batch_size, hidden_size)
        cell: tensor
            shape: (num_layers, batch_size, hidden_size)
        """
        trace_outputs, _ = self.trace_lstm(trace_prefix)
        # trace_outputs shape: (batch_size, prefix_len, hidden_size (*2 if bidirectional))
        trace_outputs = trace_outputs[:, -1, :]
        # trace_outputs shape: (batch_size, hidden_size (*2 if bidirectional))

        log_outputs, _ = self.log_lstm(log_prefix)
        # log_outputs shape: (batch_size, prefix_len, hidden_size (*2 if bidirectional))
        log_outputs = log_outputs[:, -1, :]
        # trace_outputs shape: (batch_size, hidden_size (*2 if bidirectional))

        outputs = torch.cat((trace_outputs, log_outputs), -1)
        # outputs shape: (batch_size, hidden_size*2 (*2 if bidirectional))

        # nn.Linear: input shape: (*, input_size); output shape: (*, output_size),
        # where * means any number of dimensions including none
        act_prediction = self.act_fc(outputs) 
        time_prediction = self.time_fc(outputs)
        # act_prediction shape: (batch_size, num_act)
        # time_prediction shape: (batch_size, 1)

        return act_prediction, time_prediction