"""
This module contains a class to build SEP-LSTM model

Classes:

    SEP_LSTM
     
"""

import torch
import torch.nn as nn

class SEP_LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, dropout, bidirect,
                 act_output_size, time_output_size):
        
        super(SEP_LSTM, self).__init__()
        
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
        outputs, _ = self.lstm(trace_prefix) # shape: (batch_size, prefix_len, hidden_size (*2 if bidirectional))
        outputs = outputs[:, -1, :] # shape: (batch_size, hidden_size (*2 if bidirectional))

        act_prediction = self.act_fc(outputs) # shape: (batch_size, num_act)
        time_prediction = self.time_fc(outputs) # shape: (batch_size, 1)

        return act_prediction, time_prediction
