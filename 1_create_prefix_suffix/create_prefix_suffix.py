import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from train_test_split import get_train_test_split_point


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
    
    Create log prefix for certain case list.

    Parameters
    ----------
    df: pandas.DataFrame
        Event log
    log_prefix_len: int
        Length of max log prefix
    case_list: list
        List of training/validation/test cases.
    start_idx: int
        For test set, this should be train_test_split_idx. For training/validation
        set, this should be 0.
    end_idx: int
        For test set, this should be the index of next event following 
        end_timestamp.For training/validation set, this should be train_test_split_idx.
    num_act: int
        Number of activity labels (including padding, SOC, EOC, unknown label)
    log_col_name: list
        Name(s) of column(s) containing  features.
    categorical_features: list
        Name(s) of column(s) containing categorical features.
    case_id: str
        Name of column containing case ID
    event_name: str
        Name of column containing activity label
    event_idx: str
        Index of event

    Returns
    -------
    log_prefix_cat_tensor: tensor
        shape: (num_obs, log_prefix_len, num_act + 1)

    """
    
    # ensure df is sorted by event_idx
    df = df.sort_values(by=event_idx).reset_index(drop=True)

    # create an empty list to store tensors of different col_name
    tensors_list = []

    for col in log_col_name:

        # create an empty list to store all predix tensors pertaining to one col_name
        log_prefix_list = []

        # distinguish between categorical features and continuous features.
        masking_number = int(0) if col in categorical_features else float(-10000)

        for i in range(start_idx, end_idx):
            
            # will not generate prefix ending with EOC (but will generate prefix that ending with SOC)  
            if df[case_id].iloc[i] in case_list and df[event_name].iloc[i] != 3:

                start = max(0, i - log_prefix_len + 1)
                
                prefix = df[col].iloc[start:i+1].tolist() # Prefix includes the current event

                masking = [masking_number] * max(0, log_prefix_len - len(prefix)) # make sure not to multiply a negative number

                # apply left masking
                prefix = masking + prefix
                log_prefix_list.append(prefix)

        # create tensor for each col_name    
        log_prefix_tensor = torch.tensor(log_prefix_list)

        if col in categorical_features:
            # for categorical features, one-hot encoding will be applied
            log_prefix_tensor = log_prefix_tensor.long()
            log_prefix_tensor = F.one_hot(log_prefix_tensor, num_classes=num_act)
            # log_prefix_tensor shape: (num_obs, prefix_len, num_act)
            log_prefix_tensor[:, :, 0] = 0 # To ensure that 0 padding will be encoded as all 0s
        else:
            log_prefix_tensor = log_prefix_tensor.float() 
            # log_prefix_tensor shape: (num_obs, prefix_len)
            log_prefix_tensor = log_prefix_tensor.unsqueeze(2) 
            # log_prefix_tensor shape: (num_obs, prefix_len, 1)

        # list of tensors for all col_name
        tensors_list.append(log_prefix_tensor)

    # concatenate list of tensors by the last dimension.
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
    
    Create trace prefix. Can choose between create for training set or create for test set. 

    Parameters
    ----------
    df: pandas.DataFrame
        Event log
    trace_prefix_len: integer
        Length of max trace prefix
    case_list: list
        List of training/validation/test cases.
    start_idx: int
        For test set, this should be train_test_split_idx. For training/validation
        set, this should be 0.
    end_idx: int
        For test set, this should be the index of next event following 
        end_timestamp.For training/validation set, this should be train_test_split_idx.
    num_act: int
        Number of activity labels (including padding, SOC, EOC, unknown label)
    end_timestamp: pandas.Timestamp
        The timestamp of the last event in test set after debiasing the end of 
        test set.
    col_name: list, optional
        Name(s) of column(s) containing  features.
    categorical_features: list, optional
        Name(s) of column(s) containing categorical features.
    event_idx: str
        Idex of event    
    case_id: str, optional
        Name of column containing case ID
    timestamp: str, optional
        Name of column containing timestamp
    event_name: str, optional
        Name of column containing activity label
    event_idx: str
        Index of event.   

    Returns
    -------
    trace_prefix_cat_tensor: tensor
        shape: (num_obs, trace_prefix_len, num_act + 2)

    """
    # df = df[df[case_id].isin(case_list)].copy()
    # df.reset_index(drop=True, inplace=True)

    # ensure df is sorted by event_idx
    df = df.sort_values(by=event_idx).reset_index(drop=True)
    
    # create an empty list to store tensors of different col_name
    tensors_list = []

    for col in trace_col_name:

        # create an empty list to store all prefix tensors pertaining to one col_name
        trace_prefix_list = []

        # distinguish between categorical features and continuous features.
        masking_number = int(0) if col in categorical_features else float(-10000)

        for i in range(start_idx, end_idx):
            # will not generate prefix that ends with EOC (but will generate prefix that ends with SOC)
            if df[case_id].iloc[i] in case_list and df[event_name].iloc[i] != 3:
                # get the event idex of the current event
                current_event_idx = df[event_idx].iloc[i]
                # filter the dataframe to contains rows with the same case ID as the current event
                filtered_df = df[df[case_id] == df[case_id].iloc[i]].sort_values(by=event_idx)
                # the rows before the current event in the filtered dataframe will be the prefix
                prefix = filtered_df[filtered_df[event_idx] <= current_event_idx][col].tolist() # Prefix includes the current event
                # restrict the length to trace_prefix_length
                if len(prefix) > trace_prefix_len:
                    prefix = prefix[-trace_prefix_len:]
                masking = [masking_number] * max(0, trace_prefix_len - len(prefix)) # make sure not to multiply a negative number
                # apply left masking
                prefix = masking + prefix
                trace_prefix_list.append(prefix)

        # create tensor for each col_name    
        trace_prefix_tensor = torch.tensor(trace_prefix_list)

        if col in categorical_features:
            # for categorical features, one-hot encoding will be applied
            trace_prefix_tensor = trace_prefix_tensor.long()
            trace_prefix_tensor = F.one_hot(trace_prefix_tensor, num_classes=num_act)
            trace_prefix_tensor[:, :, 0] = 0
        else:
            trace_prefix_tensor = trace_prefix_tensor.float()
            trace_prefix_tensor = trace_prefix_tensor.unsqueeze(2)

        # list of tensors for all col_name
        tensors_list.append(trace_prefix_tensor)

    # concatenate list of tensors by the last dimension.
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
    
    Create trace prefix. Can choose between create for training set or create for test set. 

    Parameters
    ----------
    df: pandas.DataFrame
        Event log
    trace_suffix_len: int
        Length of max trace suffix
    case_list: list
        List of training/validation/test cases.
    start_idx: int
        For test set, this should be train_test_split_idx. For training/validation
        set, this should be 0.
    end_idx: int
        For test set, this should be the index of next event following 
        end_timestamp.For training/validation set, this should be train_test_split_idx.
    trace_col_name: list
        Name(s) of column(s) containing  features.
    categorical_features: list
        Name(s) of column(s) containing categorical features. 
    case_id: str
        Name of column containing case ID
    event_name: str
        Name of column containing activity label  
    event_idx: str
        Idex of event   

    Returns
    -------
    suffix_tensors_list: list
        shape of each tensor in the list: (num_obs, trace_suffix_len)

    """
    # df = df[df[case_id].isin(case_list)].copy()
    # df.reset_index(drop=True, inplace=True)

    # ensure df is sorted by event_idx
    df = df.sort_values(by=event_idx).reset_index(drop=True)

    # create an empty list to store tensors of different col_name
    suffix_tensors_list = []

    for col in trace_col_name:

        # create an empty list to store all suffix tensors pertaining to one col_name
        trace_suffix_list = []

        # distinguish between categorical features and continuous features.
        masking_number = int(0) if col in categorical_features else float(-10000)

        for i in range(start_idx, end_idx):
            # will not generate prefix that ends with EOC (but will generate prefix that ends with SOC)
            if df[case_id].iloc[i] in case_list and df[event_name].iloc[i] != 3:
                # get the event idex of the current event
                current_event_idx = df[event_idx].iloc[i]
                # filter the dataframe to contains rows with the same case ID as the current event
                filtered_df = df[df[case_id] == df[case_id].iloc[i]].sort_values(by=event_idx)
                # the rows after the current event in the filtered dataframe will be the suffix
                suffix = filtered_df[filtered_df[event_idx] > current_event_idx][col].tolist()
                # restrict the length to trace_suffix_length
                if len(suffix) > trace_suffix_len:
                    suffix = suffix[:trace_suffix_len]
                masking = [masking_number] * max(0, trace_suffix_len - len(suffix)) # make sure that (trace_suffix_length - len(suffix)) would not be a negative number
                # apply right masking
                suffix = suffix + masking
                trace_suffix_list.append(suffix)
        # create tensor for each col_name    
        trace_suffix_tensor = torch.tensor(trace_suffix_list)

        # for categorical features, the type of the tensor should be long (for softmax)
        if col in categorical_features:
            trace_suffix_tensor = trace_suffix_tensor.long() # this is a 2D tensor (num_bs * seq_length)
        else:
            trace_suffix_tensor = trace_suffix_tensor.float()  # this is a 2D tensor (num_bs * seq_length)

        # list of tensors for all col_name
        suffix_tensors_list.append(trace_suffix_tensor)

    return suffix_tensors_list

def create_log_next(df,
                    case_list,
                    start_idx,
                    end_idx,
                    log_col_name,
                    categorical_features,
                    case_id, 
                    event_name,
                    event_idx):
    """
    
    Create log prefix for certain case list.

    Parameters
    ----------
    df: pandas.DataFrame
        Event log
    case_list: list
        List of training/validation/test cases.
    start_idx: int
        For test set, this should be train_test_split_idx. For training/validation
        set, this should be 0.
    end_idx: int
        For test set, this should be the index of next event following 
        end_timestamp.For training/validation set, this should be train_test_split_idx.
    log_col_name: list
        Name(s) of column(s) containing  features.
    categorical_features: list
        Name(s) of column(s) containing categorical features.
    case_id: str
        Name of column containing case ID
    event_name: str
        Name of column containing activity label
    event_idx: str
        Index of event

    Returns
    -------
    tensors_list: list
        shape of each tensor in the list: (num_obs)

    """
    
    # ensure df is sorted by event_idx
    df = df.sort_values(by=event_idx).reset_index(drop=True)

    # create an empty list to store tensors of different col_name
    tensors_list = []

    for col in log_col_name:

        # create an empty list to store all tensors pertaining to one col_name
        log_next_list = []

        for i in range(start_idx, end_idx):
            
            if df[case_id].iloc[i] in case_list and df[event_name].iloc[i] != 3:

                next = df[col].iloc[i+1]
            
                log_next_list.append(next)

        # create tensor for each col_name    
        log_next_tensor = torch.tensor(log_next_list)

        if col in categorical_features:
            log_next_tensor = log_next_tensor.long()
        else:
            log_next_tensor = log_next_tensor.float() 

        # list of tensors for all col_name
        tensors_list.append(log_next_tensor)

    return tensors_list
