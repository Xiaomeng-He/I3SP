"""
This module contains functions and a class for training and evaluating a 
Seq2Seq model.

Functions:
    train
    validate
    evaluate
    loss_function
    damerau_levenshtein
    performance_metrics
    init_weights

Class:
    EarlyStopper   
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

def train(model, 
          dataloader,
          lr,
          tf_rate,
          class_weights,
          device):
    """ 
    Train Seq2Seq model.

    Parameters
    ----------
    model : torch.nn.Module
        An instance of the Seq2Seq_trace or Seq2Seq_cat class.
    dataloader : torch.utils.data.DataLoader
        Dataloader containing the training set data.
    lr : float
        Learning rate.
    tf_rate : float
        Teacher forcing ratio, representing the probability of using the 
        expected output rather than the predicted output from the previous 
        step as input. A tf_rate of 1 means strict teacher forcing; 0 means no 
        teacher forcing.
    class_weights : torch.Tensor
        Weights assigned to different activity labels when calculating 
        cross-entropy loss.  
        Shape: (num_act,)
    device : torch.device
        Computation device (GPU or CPU).

    Returns
    -------
    avg_train_loss : float
        Total training loss, computed as the unweighted sum of the activity label 
        and timestamp suffix prediction losses.
    avg_train_act_loss : float
        Training loss for activity label suffix prediction.
    avg_train_time_loss : float
        Training loss for timestamp suffix prediction.
    """
    model.train() 

    # define optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # initialize losses
    train_epoch_loss, train_epoch_act_loss, train_epoch_time_loss = 0.0, 0.0, 0.0

    for batch in dataloader:

        # load data
        train_log_prefix, train_trace_prefix, train_act_suffix, train_time_suffix = batch
        # train_log_prefix shape: (batch_size, log_prefix_len, num_act + 1)
        # train_trace_prefix shape: (batch_size, trace_prefix_len, num_act + 2)
        # train_act_suffix shape: (batch_size, suffix_len)
        # train_time_suffix shape: (batch_size, suffix_len)
        
        train_log_prefix = train_log_prefix.float().to(device)
        train_trace_prefix = train_trace_prefix.float().to(device)
        train_time_suffix = train_time_suffix.float().to(device)
        train_act_suffix = train_act_suffix.long().to(device)

        # set the gradient to zero
        optimizer.zero_grad()

        # run a forward pass and obtain predictions
        act_predictions, time_predictions = model(train_log_prefix,
                                                  train_trace_prefix, 
                                                  train_act_suffix, 
                                                  train_time_suffix, 
                                                  tf_rate)
        # act_predictions shape: (batch_size, suffix_len, num_act)
        # time_predictions shape: (batch_size, suffix_len, 1)

        # ensure predictions are stored in device
        act_predictions, time_predictions = act_predictions.to(device), time_predictions.to(device)
        
        # calculate act_loss, time_loss and loss
        loss, act_loss, time_loss = loss_function(act_predictions,
                                                 time_predictions,
                                                 train_act_suffix,
                                                 train_time_suffix,
                                                 class_weights,
                                                 device)

        # backpropagation
        loss.backward()
        optimizer.step()

        # sum up losses from all batches
        train_epoch_loss += loss.item()
        train_epoch_act_loss += act_loss.item()
        train_epoch_time_loss += time_loss.item()
    
    # compute average losses over the batch
    avg_train_loss = train_epoch_loss / len(dataloader)
    avg_train_act_loss = train_epoch_act_loss / len(dataloader)
    avg_train_time_loss = train_epoch_time_loss / len(dataloader)
    
    return avg_train_loss, avg_train_act_loss, avg_train_time_loss

def validate(model,
            dataloader,
            device,
            max_value,
            min_value,
            class_weights,
            tf_rate=0):
    """
    Calculate loss and performance metrics on the validation set.

    Parameters
    ----------
    model : torch.nn.Module
        An instance of the Seq2Seq_trace or Seq2Seq_cat class.
    dataloader : torch.utils.data.DataLoader
        Dataloader containing the validation set data.
    device : torch.device
        Computation device (GPU or CPU).
    max_value : float
        Maximum of the log-normalized TTNE.
    min_value : float
        Minimum of the log-normalized TTNE.
    class_weights : torch.Tensor
        Weights assigned to different activity labels when calculating 
        cross-entropy loss.  
        Shape: (num_act,)
    tf_rate : float
        Teacher forcing ratio, representing the probability of using the 
        expected output rather than the predicted output from the previous 
        step as input. A tf_rate of 1 means strict teacher forcing; 0 means no 
        teacher forcing.
        Default is 0.

    Returns
    -------
    avg_val_loss : float
        Total validation loss, computed as the unweighted sum of the activity 
        label and timestamp suffix prediction losses.
    avg_val_act_loss : float
        Validation loss for activity label suffix prediction.
    avg_val_time_loss : float
        Validation loss for timestamp suffix prediction.
    avg_val_dl_distance : float
        Normalized Damerau-Levenshtein distance for activity label suffix 
        prediction on the validation set.
    avg_val_mae : float
        Mean Absolute Error (MAE) for timestamp suffix prediction on the 
        validation set.
    avg_val_msle : float
        Mean Squared Logarithmic Error (MSLE) for timestamp suffix prediction on 
        the validation set.
    """
    model.eval()

    # initialize losses and performance metrics
    val_epoch_loss, val_epoch_act_loss, val_epoch_time_loss = 0.0, 0.0, 0.0
    val_epoch_dl_distance, val_epoch_mae, val_epoch_msle = 0.0, 0.0, 0.0

    with torch.no_grad():

        for batch in dataloader:

            # load data
            log_prefix, trace_prefix, act_suffix, time_suffix = batch
            # log_prefix shape: (batch_size, log_prefix_len, num_act + 1)
            # trace_prefix shape: (batch_size, trace_prefix_len, num_act + 2)
            # act_suffix shape: (batch_size, suffix_len)
            # time_suffix shape: (batch_size, suffix_len)

            log_prefix = log_prefix.float().to(device)
            trace_prefix = trace_prefix.float().to(device)
            time_suffix = time_suffix.float().to(device)
            act_suffix = act_suffix.to(torch.long).to(device)

            # run a forward pass and obtain predictions
            act_predictions, time_predictions = model(log_prefix,
                                                      trace_prefix, 
                                                      act_suffix, 
                                                      time_suffix, 
                                                      tf_rate)
            # act_predictions shape: (batch_size, suffix_len, num_act)
            # time_predictions shape: (batch_size, suffix_len, 1)

            # ensure predictions are stored in device
            act_predictions, time_predictions = act_predictions.to(device), time_predictions.to(device)

            # calculate losses using loss function
            val_loss, val_act_loss, val_time_loss = loss_function(act_predictions,
                                                                  time_predictions,
                                                                  act_suffix,
                                                                  time_suffix,
                                                                  class_weights,
                                                                  device)
            
            # calculate performance metrics
            dl_distance, mae, msle = performance_metrics(act_predictions,
                                                         time_predictions,
                                                         act_suffix,
                                                         time_suffix,
                                                         max_value,
                                                         min_value)
            
            # sum up losses and metrics from all batches
            val_epoch_loss += val_loss.item()
            val_epoch_act_loss += val_act_loss.item()
            val_epoch_time_loss += val_time_loss.item()
            val_epoch_dl_distance += dl_distance
            val_epoch_mae += mae
            val_epoch_msle += msle

    # compute average losses and metrics over the batch
    avg_val_loss = val_epoch_loss / len(dataloader)
    avg_val_act_loss = val_epoch_act_loss / len(dataloader)
    avg_val_time_loss = val_epoch_time_loss / len(dataloader)
    avg_val_dl_distance = val_epoch_dl_distance / len(dataloader)
    avg_val_mae = val_epoch_mae / len(dataloader)
    avg_val_msle = val_epoch_msle / len(dataloader)
    
    return avg_val_loss, avg_val_act_loss, avg_val_time_loss, \
        avg_val_dl_distance, avg_val_mae, avg_val_msle

def evaluate(model,
            dataloader,
            device,
            max_value,
            min_value,
            tf_rate=0):
    """
    Calculate performance metrics on the test set.

    Parameters
    ----------
    model : torch.nn.Module
        An instance of the Seq2Seq_trace or Seq2Seq_cat class.
    dataloader : torch.utils.data.DataLoader
        Dataloader containing the test set data.
    device : torch.device
        Computation device (GPU or CPU).
    max_value : float
        Maximum of the log-normalized TTNE.
    min_value : float
        Minimum of the log-normalized TTNE.
    tf_rate : float
        Teacher forcing ratio, representing the probability of using the 
        expected output rather than the predicted output from the previous 
        step as input. A tf_rate of 1 means strict teacher forcing; 0 means no 
        teacher forcing.
        Default is 0.

    Returns
    -------
    avg_test_dl_distance : float
        Normalized Damerau-Levenshtein distance for activity label suffix 
        prediction on the test set.
    avg_test_mae : float
        Mean Absolute Error (MAE) for timestamp suffix prediction on the test
        set.
    avg_test_msle: float
        Mean Squared Logarithmic Error (MSLE) for timestamp suffix prediction on 
        the test set.
    """ 
    model.eval()

    # initialize performance metrics
    test_epoch_dl_distance, test_epoch_mae, test_epoch_msle  = 0.0, 0.0, 0.0

    with torch.no_grad():

        for batch in dataloader:

            # load data
            log_prefix, trace_prefix, act_suffix, time_suffix = batch
            # log_prefix shape: (batch_size, log_prefix_len, num_act + 1)
            # trace_prefix shape: (batch_size, trace_prefix_len, num_act + 2)
            # act_suffix shape: (batch_size, suffix_len)
            # time_suffix shape: (batch_size, suffix_len)

            log_prefix = log_prefix.float().to(device)
            trace_prefix = trace_prefix.float().to(device)
            time_suffix = time_suffix.float().to(device)
            act_suffix = act_suffix.to(torch.long).to(device)

            # run a forward pass and obtain predictions
            act_predictions, time_predictions = model(log_prefix,
                                                      trace_prefix, 
                                                      act_suffix, 
                                                      time_suffix, 
                                                      tf_rate)
            # act_predictions shape: (batch_size, suffix_len, num_act)
            # time_predictions shape: (batch_size, suffix_len, 1)

            # ensure predictions are stored in device
            act_predictions, time_predictions = act_predictions.to(device), time_predictions.to(device)
            
            # calculate performance metrics
            dl_distance, mae, msle = performance_metrics(
                act_predictions,
                time_predictions,
                act_suffix,
                time_suffix,
                max_value,
                min_value)
            
            # sum up losses from all batches
            test_epoch_dl_distance += dl_distance
            test_epoch_mae += mae
            test_epoch_msle += msle

    # compute average metrics over the batch
    avg_test_dl_distance = test_epoch_dl_distance / len(dataloader)
    avg_test_mae =  test_epoch_mae / len(dataloader)
    avg_test_msle =  test_epoch_msle / len(dataloader)
    
    return avg_test_dl_distance, avg_test_mae, avg_test_msle

def loss_function(act_predictions, 
                  time_predictions, 
                  act_suffix,
                  time_suffix,
                  class_weights,
                  device,
                  time_masking = float(-10000)):
    """
    Calculate cross-entropy loss for activity label suffix prediction and Mean 
    Absolute Error (MAE) for timestamp suffix prediction. The final loss is the 
    unweighted sum of both losses.

    Parameters
    ----------
    act_predictions : torch.Tensor
        Activity label suffix predictions.  
        Shape: (batch_size, suffix_len, num_act)
    time_predictions : torch.Tensor
        Timestamp suffix predictions.  
        Shape: (batch_size, suffix_len, 1)
    act_suffix : torch.Tensor
        Ground truth activity label suffix.  
        Shape: (batch_size, suffix_len)
    time_suffix : torch.Tensor
        Ground truth timestamp suffix.  
        Shape: (batch_size, suffix_len)
    class_weights : torch.Tensor
        Weights assigned to different activity labels when calculating 
        cross-entropy loss.  
        Shape: (num_act,)
    device : torch.device
        Computation device (GPU or CPU).
    time_masking : float
        Value representing masked entries in the timestamp suffix tensor.
        Default is -10000.

    Returns
    -------
    act_loss : torch.Tensor
        A scalar representing activity label suffix prediction loss, averaged 
        over each element in the batch.
    time_loss : torch.Tensor
        A scalar representing timestamp suffix prediction loss, averaged over 
        each element in the batch.
    loss : torch.Tensor
        A scalar representing the final loss.  
        loss = 0.5 * act_loss + 0.5 * time_loss
    """
    # define loss functions
    act_criterion = nn.CrossEntropyLoss(weight=class_weights,
                                        ignore_index=0)
    time_criterion = nn.L1Loss()

    # calculate act loss
    act_predictions = act_predictions.view(-1, act_predictions.size(-1)) # shape: (batch_size * suffix_len, num_act)
    act_suffix = act_suffix.view(-1) # shape: (batch_size * suffix_len,)
    act_loss = act_criterion(act_predictions, act_suffix)

    # calculate time loss
    time_suffix = time_suffix.unsqueeze(-1) # shape: (batch_size, suffix_len, 1) 
    # mask padded entries (-10000) in the timestamp suffix so that they do not 
    # contribute to the gradient
    mask = (time_suffix != time_masking).to(device)
    masked_time_suffix = torch.masked_select(time_suffix, mask) 
    masked_time_predictions = torch.masked_select(time_predictions, mask)
    time_loss = time_criterion(masked_time_predictions, masked_time_suffix)

    # calculate overall loss
    loss = 0.5 * act_loss + 0.5 * time_loss

    return loss, act_loss, time_loss

def damerau_levenshtein(list1, 
                        list2):
    """
    Compute the Damerau-Levenshtein distance between two sequences of activity labels.

    The Damerau-Levenshtein distance measures the minimum number of operations 
    (insertions, deletions, substitutions, and transpositions of adjacent elements) 
    required to transform one list into the other.

    Parameters
    ----------
    list1 : list
        First sequence of activity labels.
    list2 : list
        Second sequence of activity labels.

    Returns
    -------
    dl_distance : float
        The computed Damerau-Levenshtein distance between the two sequences.
    """
    len_1, len_2 = len(list1), len(list2)

    dist = [[0 for _ in range(len_2 + 1)] for _ in range(len_1 + 1)]

    for i in range(len_1 + 1):
        dist[i][0] = i
    for j in range(len_2 + 1):
        dist[0][j] = j

    for i in range(1, len_1 + 1):
        for j in range(1, len_2 + 1):
            cost = 0 if list1[i - 1] == list2[j - 1] else 1

            dist[i][j] = min(
                dist[i - 1][j] + 1,    # deletion
                dist[i][j - 1] + 1,    # insertion
                dist[i - 1][j - 1] + cost  # substitution
            )

            if i > 1 and j > 1 and list1[i - 1] == list2[j - 2] and list1[i - 2] == list2[j - 1]:
                dist[i][j] = min(
                    dist[i][j],
                    dist[i - 2][j - 2] + cost  # transposition
                )

    dl_distance = dist[len_1][len_2]

    return dl_distance

def performance_metrics(act_predictions, 
                        time_predictions, 
                        act_suffix,
                        time_suffix,
                        max_value,
                        min_value,
                        eoc_index = int(3)):
    """
    Compute three performance metrics for suffix prediction:  
    - dl_distance: Normalized Damerau-Levenshtein distance between predicted and 
      ground truth activity label suffixes.
    - mae: Mean Absolute Error between predicted and ground truth TTNE suffixes.
    - msle: Mean Squared Logarithmic Error between predicted and ground truth 
      TTNE suffixes.
    
    Parameters
    ----------
    act_predictions : torch.Tensor
        Activity label suffix predictions.  
        Shape: (batch_size, suffix_len, num_act)
    time_predictions : torch.Tensor
        Timestamp suffix predictions.  
        Shape: (batch_size, suffix_len, 1)
    act_suffix : torch.Tensor
        Ground truth activity label suffix.  
        Shape: (batch_size, suffix_len)
    time_suffix : torch.Tensor
        Ground truth timestamp suffix.  
        Shape: (batch_size, suffix_len)
    max_value : float
        Maximum of the log-normalized TTNE.
    min_value : float
        Minimum of the log-normalized TTNE.
    eoc_index : int
        Index representing the End-Of-Case (EOC) token in the activity label 
        vocabulary. 
        Default is 3.
    
    Returns
    -------
    dl_distance : float
        Average normalized Damerau-Levenshtein distance over the batch.   
    mae : float
        Average MAE over the batch.
    msle : float
        Average MSLE over the batch.
    """
    # get batch_size
    batch_size = act_predictions.shape[0]

    # convert predicted activity label probabilities to lebel indices
    act_predictions = act_predictions.argmax(2) # shape: (batch_size, suffix_len)

    # reshape time_predictions
    time_predictions = time_predictions.squeeze(-1) # shape: (batch_size, suffix_len)

    # de-normalize and reverse log transformation of TTNE
    time_suffix = time_suffix * (max_value - min_value) + min_value
    time_suffix_exp = torch.exp(time_suffix) - 1   
    time_predictions = time_predictions * (max_value - min_value) + min_value
    time_predictions_exp = torch.exp(time_predictions) - 1

    # initialize performance metrics
    total_dl_distance, total_mae, total_msle = 0.0, 0.0, 0.0

    # define loss functions
    time_criterion_1 = nn.L1Loss()
    time_criterion_2 = nn.MSELoss()

    for i in range(batch_size):

        # -- normalized Damerau-Levenshtein distance --

        # convert predicted and target activity label sequences to lists, 
        # truncated at EOC
        act_pred_list = []
        act_target_list = []

        for p in act_predictions[i].tolist():
            act_pred_list.append(p)
            if p == eoc_index: # stop at EOC token (inclusive)
                break    
            
        for t in act_suffix[i].tolist():
            act_target_list.append(t)
            if t == eoc_index: # stop at EOC token (inclusive)
                break
        
        # compute normalized Damerau-Levenshtein distance
        pred_len, target_len= len(act_pred_list), len(act_target_list)
        max_len = max(pred_len, target_len)
        assert max_len > 0, "Error: max_len should be greater than 0."

        distance = damerau_levenshtein(act_pred_list, act_target_list)
        normalized_distance = distance / max_len
        total_dl_distance += normalized_distance

        # -- MAE --

        time_target_exp = time_suffix_exp[i, :target_len] # shape: (target_len,)
        time_pred_exp = time_predictions_exp[i, :target_len] # shape: (target_len,)
        mae_loss = time_criterion_1(time_target_exp, time_pred_exp)
        total_mae +=  mae_loss.item()

        # -- MSLE --

        time_target = time_suffix[i, :target_len] # shape: (target_len,)
        time_pred = time_predictions[i, :target_len] # shape: (target_len,)
        msle_loss = time_criterion_2(time_target, time_pred)
        total_msle +=  msle_loss.item() 

    # compute average metrics over the batch
    dl_distance = total_dl_distance / batch_size
    mae = total_mae / batch_size
    msle = total_msle / batch_size
    
    return dl_distance, mae, msle

class EarlyStopper:
    """
    Implements early stopping to terminate training when validation performance 
    stops improving.

    Inspired by: https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch

    Parameters
    ----------
    patience : int
        Number of epochs with no significant improvement in validation metrics 
        before stopping.

    Attributes
    ----------
    patience : int
        The number of epochs with no significant improvement in validation 
        metrics before stopping.
    counter : int
        The current number of consecutive epochs without improvement.
    min_val_dl_distance : float
        The minimum normalized Damerau-Levenshtein distance observed on the 
        validation set.
    min_val_mae_padboth : float
        The minimum MAE observed on the validation set.
    min_val_loss : float
        The minimum validation loss observed.

    Methods
    -------
    early_stop(val_dl_distance, val_mae, val_loss, dl_distance=0.001, mae=500, loss=0.001) -> bool
        Checks if training should be stopped.
    """
    def __init__(self, patience):
        self.patience = patience 
        self.counter = 0 
        self.min_val_dl_distance = float('inf')
        self.min_val_mae_padboth = float('inf')
        self.min_val_loss = float('inf')

    def early_stop(self, 
                   val_dl_distance, 
                   val_mae, 
                   val_loss, 
                   dl_distance = 0.001, 
                   mae = 500, 
                   loss = 0.001):
        """
        Determine whether to stop training early based on validation loss and 
        metrics.

        Parameters
        ----------
        val_dl_distance : float
            Current epoch's normalized Damerau-Levenshtein distance on the
            validation set.
        val_mae : float
            Current epoch's MAE on the validation set.
        val_loss : float
            Current epoch's validation loss.
        dl_distance : float
            Minimum improvement in normalized Damerau-Levenshtein distanc to 
            reset the counter. 
            Default is 0.001.
        mae : float
            Minimum improvement in MAE to reset the counter. 
            Default is 500.
        loss : float
            Minimum improvement in validation loss to reset the counter. 
            Default is 0.001.

        Returns
        -------
        bool
            True if training should stop early, False otherwise.
        """
        improvement = False

        if val_dl_distance < (self.min_val_dl_distance - dl_distance):
            self.min_val_dl_distance = val_dl_distance
            improvement = True

        if val_mae < (self.min_val_mae - mae):
            self.min_val_mae_padboth = val_mae
            improvement = True
        
        if val_loss < (self.min_val_loss - loss):
            self.min_val_loss = val_loss
            improvement = True
        
        if improvement:
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            return True
        
        return False

def init_weights(m):
    """
    Initialize model parameters uniformly in the range [-0.08, 0.08].

    Parameters
    ----------
    m : torch.nn.Module
        An instance of the Seq2Seq_trace or Seq2Seq_cat class.
    """
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
