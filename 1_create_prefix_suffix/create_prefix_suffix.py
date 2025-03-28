"""
This module contains functions to generate log prefix, trace prefix, and trace 
suffix as tensors.

Functions:
    create_log_prefix
    create_trace_prefix
    create_trace_suffix   
"""

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

def create_log_prefix(df, 
                      log_prefix_len, 
                      case_list,
                      start_idx,
                      end_idx,
                      num_act,
                      log_col_name,
                      categorical_features,
                      case_id, 
                      event_name,
                      event_idx):
    """
    Create log prefix tensor.

    Parameters
    ----------
    df : pandas.DataFrame
        Event log.
    log_prefix_len : int
        The window size of the log prefix.
    case_list : list
        List of cases (training/validation/test cases); only events in these 
        cases are used to generate the log prefix. 
    start_idx: int
        Index of the start of the range (inclusive)
        For the test set, this should be train_test_split_idx. 
        For training/validation set, this should be 0.
    end_idx : int
        Index of the end of the range (exclusive). Prefixes are generated for 
        events from start_idx up to, but not including, end_idx.
        For the test set, this should be the index of the first event after 
        end_timestamp.
        For training/validation set, this should be train_test_split_idx.
    num_act : int
        Number of activity labels (including padding, SOC, EOC, unknown label).
    log_col_name : list
        Name(s) of column(s) containing features used in the log prefix.
    categorical_features : list
        Name(s) of column(s) containing categorical features.
    case_id : str
        Name of the column containing case IDs.
    event_name : str
        Name of the column containing activity labels.
    event_idx : str
        Name of the column containing event ordering information.

    Returns
    -------
    log_prefix_cat_tensor : tensor
        A tensor storing log prefixes.
        Shape: (num_samples, log_prefix_len, num_act + 1)
    """
    # ensure event log is sorted by event_idx
    df = df.sort_values(by=event_idx).reset_index(drop=True)

    tensors_list = []

    for col in log_col_name:

        log_prefix_list = []

        # set masking values for categorical and continuous features
        masking_number = int(0) if col in categorical_features else float(-10000)

        for i in range(start_idx, end_idx):
            
            # skip EOC events (event_name == 3), but include SOC 
            if df[case_id].iloc[i] in case_list and df[event_name].iloc[i] != 3:

                start = max(0, i - log_prefix_len + 1)                
                prefix = df[col].iloc[start:i+1].tolist()                

                # apply left masking
                masking = [masking_number] * max(0, log_prefix_len - len(prefix)) 
                prefix = masking + prefix
                log_prefix_list.append(prefix)

        # create tensor for each feature column
        log_prefix_tensor = torch.tensor(log_prefix_list)

        if col in categorical_features:
            # one-hot encode categorical features
            log_prefix_tensor = log_prefix_tensor.long()
            log_prefix_tensor = F.one_hot(log_prefix_tensor, num_classes=num_act)
            log_prefix_tensor[:, :, 0] = 0
        else:
            log_prefix_tensor = log_prefix_tensor.float() 
            log_prefix_tensor = log_prefix_tensor.unsqueeze(2) 

        tensors_list.append(log_prefix_tensor)

    log_prefix_cat_tensor = torch.cat(tensors_list, dim=2)

    return log_prefix_cat_tensor

def create_trace_prefix(df, 
                        trace_prefix_len, 
                        case_list,
                        start_idx,
                        end_idx,
                        num_act,
                        trace_col_name,
                        categorical_features,
                        case_id,
                        event_name,
                        event_idx):
    """
    Create trace prefix tensor.

    Parameters
    ----------
    df : pandas.DataFrame
        Event log.
    trace_prefix_len : int
        The maximum length of the trace prefix.
    case_list : list
        List of cases (training/validation/test cases); only events in these 
        cases are used to generate the trace prefix.
    start_idx : int
        Index of the start of the range (inclusive).
        For the test set, this should be train_test_split_idx.
        For training/validation set, this should be 0.
    end_idx : int
        Index of the end of the range (exclusive). Prefixes are generated for 
        events from start_idx up to, but not including, end_idx.
        For the test set, this should be the index of the first event after 
        end_timestamp.
        For training/validation set, this should be train_test_split_idx.
    num_act : int
        Number of activity labels (including padding, SOC, EOC, unknown label).
    trace_col_name : list
        Name(s) of column(s) containing features used in the trace prefix.
    categorical_features : list
        Name(s) of column(s) containing categorical features.
    case_id : str
        Name of the column containing case IDs.
    event_name : str
        Name of the column containing activity labels.
    event_idx : str
        Name of the column containing event ordering information.

    Returns
    -------
    trace_prefix_cat_tensor : tensor
        A tensor storing trace prefixes.
        Shape: (num_samples, trace_prefix_len, num_act + 1)
    """

    # ensure event log is sorted by event_idx
    df = df.sort_values(by=event_idx).reset_index(drop=True)
    
    tensors_list = []

    for col in trace_col_name:

        trace_prefix_list = []

        # set masking values for categorical and continuous features
        masking_number = int(0) if col in categorical_features else float(-10000)

        for i in range(start_idx, end_idx):

            # skip EOC events (event_name == 3), but include SOC
            if df[case_id].iloc[i] in case_list and df[event_name].iloc[i] != 3:
 
                current_event_idx = df[event_idx].iloc[i]

                # get all events of the current case
                filtered_df = df[df[case_id] == df[case_id].iloc[i]].sort_values(by=event_idx)
                
                # get events up to and including the current event
                prefix = filtered_df[filtered_df[event_idx] <= current_event_idx][col].tolist() 
                
                # restrict prefix length to trace_prefix_len
                if len(prefix) > trace_prefix_len:
                    prefix = prefix[-trace_prefix_len:]

                # apply left masking
                masking = [masking_number] * max(0, trace_prefix_len - len(prefix)) 
                prefix = masking + prefix
                trace_prefix_list.append(prefix)

        # create tensor for each feature column
        trace_prefix_tensor = torch.tensor(trace_prefix_list)

        if col in categorical_features:
            # one-hot encode categorical features
            trace_prefix_tensor = trace_prefix_tensor.long()
            trace_prefix_tensor = F.one_hot(trace_prefix_tensor, num_classes=num_act)
            trace_prefix_tensor[:, :, 0] = 0
        else:
            trace_prefix_tensor = trace_prefix_tensor.float()
            trace_prefix_tensor = trace_prefix_tensor.unsqueeze(2)

        tensors_list.append(trace_prefix_tensor)

    trace_prefix_cat_tensor = torch.cat(tensors_list, dim=2)

    return trace_prefix_cat_tensor

def create_trace_suffix(df, 
                        trace_suffix_len,
                        case_list, 
                        start_idx,
                        end_idx,
                        trace_col_name,
                        categorical_features,
                        case_id, 
                        event_name,
                        event_idx):
    """
    Create trace suffix tensor.

    Parameters
    ----------
    df : pandas.DataFrame
        Event log.
    trace_suffix_len : int
        The maximum length of the trace suffix.
    case_list : list
        List of cases (training/validation/test cases); only events in these 
        cases are used to generate the trace suffix.
    start_idx : int
        Index of the start of the range (inclusive).
        For the test set, this should be train_test_split_idx.
        For training/validation set, this should be 0.
    end_idx : int
        Index of the end of the range (exclusive). Suffixes are generated for 
        events from start_idx up to, but not including, end_idx.
        For the test set, this should be the index of the first event after 
        end_timestamp.
        For training/validation set, this should be train_test_split_idx.
    trace_col_name : list
        Name(s) of column(s) containing features used in the trace suffix.
    categorical_features : list
        Name(s) of column(s) containing categorical features.
    case_id : str
        Name of the column containing case IDs.
    event_name : str
        Name of the column containing activity labels.
    event_idx : str
        Name of the column containing event ordering information.

    Returns
    -------
    suffix_tensors_list : list of torch.Tensor
        Each tensor has shape (num_samples, trace_suffix_len). One tensor is 
        returned per feature column.
    """

    # ensure event log is sorted by event_idx
    df = df.sort_values(by=event_idx).reset_index(drop=True)

    suffix_tensors_list = []

    for col in trace_col_name:

        trace_suffix_list = []

        # set masking values for categorical and continuous features
        masking_number = int(0) if col in categorical_features else float(-10000)

        for i in range(start_idx, end_idx):
            # skip EOC events (event_name == 3), but include SOC
            if df[case_id].iloc[i] in case_list and df[event_name].iloc[i] != 3:

                current_event_idx = df[event_idx].iloc[i]
                # get all events of the current case
                filtered_df = df[df[case_id] == df[case_id].iloc[i]].sort_values(by=event_idx)

                # get events after the current event
                suffix = filtered_df[filtered_df[event_idx] > current_event_idx][col].tolist()
                
                # restrict suffix length to trace_suffix_len
                if len(suffix) > trace_suffix_len:
                    suffix = suffix[:trace_suffix_len]
                
                # apply right masking
                masking = [masking_number] * max(0, trace_suffix_len - len(suffix)) # make sure that (trace_suffix_length - len(suffix)) would not be a negative number
                suffix = suffix + masking
                trace_suffix_list.append(suffix)

        # create tensor for each feature column    
        trace_suffix_tensor = torch.tensor(trace_suffix_list)

        if col in categorical_features:
            trace_suffix_tensor = trace_suffix_tensor.long() 
        else:
            trace_suffix_tensor = trace_suffix_tensor.float() 

        suffix_tensors_list.append(trace_suffix_tensor)

    return suffix_tensors_list