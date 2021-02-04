import pandas as pd
import base64
import datetime
import io
import dash_html_components as html
import numpy as np
from scipy import signal

def parse_data(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV or TXT file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')), header=0)
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        elif 'txt' or 'tsv' in filename:
            # Assume that the user upl, delimiter = r'\s+'oaded an excel file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')), delimiter=r'\s+')
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return df

def is_start(x, x_pre, threshold):
    if np.abs(x) < threshold and np.abs(x_pre) >= threshold:
        return True
    return False

def is_end(x, x_after, threshold):
    if np.abs(x) < threshold and np.abs(x_after) >= threshold:
        return True
    return False

def concate_remove_index(start_list,end_list,remove_sliding_window = 0):
    start_list = np.array(start_list)
    end_list = np.array(end_list)
    diff_list = start_list[1:]-end_list[:-1]
    end_list_rm_indices = np.where(diff_list<=remove_sliding_window)[0]
    start_list_rm_indices = np.where(diff_list <= remove_sliding_window)[0]+1
    start_out_list = np.delete(start_list,start_list_rm_indices)
    end_out_list = np.delete(end_list, end_list_rm_indices)
    return start_out_list,end_out_list

def remove_short_length(start_milestone,end_milestone,min_length=500):
    remove_idx = []
    for idx in range(len(end_milestone)):
        try:
            if (end_milestone[idx] - start_milestone[idx]) < min_length:
                remove_idx.append(idx)
        except Exception as error:
            print(error)
    start_milestone = np.delete(start_milestone, remove_idx)
    end_milestone = np.delete(end_milestone, remove_idx)
    return start_milestone,end_milestone

def trim_missing_signal(df):
    indices_0 = np.array(df[df["PLETH"] == 0].index)
    indices_start_end = np.hstack((0, indices_0, len(df) - 1))
    diff_res = indices_start_end[1:] - indices_start_end[:-1]
    sequence_diff_threshold = np.median(list(set(diff_res)))
    start_cut_pivot = []
    end_cut_pivot = []
    for idx in range(1, len(diff_res) - 1):
        if is_start(diff_res[idx], diff_res[idx - 1], sequence_diff_threshold):
            start_cut_pivot.append(indices_0[idx])
        if is_end(diff_res[idx], diff_res[idx + 1], sequence_diff_threshold):
            end_cut_pivot.append(indices_0[idx])
    start_milestone,end_milestone = cut_milestone_to_keep_milestone(start_cut_pivot,end_cut_pivot,len(df))
    return start_milestone,end_milestone

def cut_milestone_to_keep_milestone(start_cut_pivot,end_cut_pivot,length_df):
    if 0 not in np.array(start_cut_pivot):
        start_milestone = np.hstack((0, np.array(end_cut_pivot) + 1))
        if length_df - 1 not in np.array(end_cut_pivot):
            end_milestone = np.hstack((np.array(start_cut_pivot) - 1, length_df - 1))
        else:
            end_milestone = (np.array(start_cut_pivot) - 1)
    else:
        start_milestone = np.array(end_cut_pivot) + 1
        end_milestone = np.hstack((np.array(start_cut_pivot)[1:] - 1, length_df - 1))
    return start_milestone,end_milestone

def trim_by_frequency_partition(df_examine,start_milestone,end_milestone,
                                window_size=500,peak_threshold_ratio=None,
                                remove_sliding_window=None):
    # df_examine = df["PLETH"].iloc[start_milestone[1]:end_milestone[1]]
    if window_size == None:
        window_size = 500
    if window_size > len(df_examine):
        window_size  = len(df_examine)
    if peak_threshold_ratio == None:
        peak_threshold_ratio = 1.8
    if remove_sliding_window == None:
        remove_sliding_window = 0
    taper_windows = signal.hanning(window_size)
    window = signal.get_window("boxcar", window_size)
    welch_full = signal.welch(df_examine, window=window)
    peaks_full = signal.find_peaks(welch_full[1], threshold=np.mean(welch_full[1]))

    remove_start_indices = []
    remove_end_indices = []

    pointer = 0
    # peak_threshold_ratio = 1.8

    overlap_rate = 1

    while pointer < len(df_examine):
        end_pointer = pointer + (window_size)
        if end_pointer >= len(df_examine):
            break
        small_partition = df_examine[pointer:end_pointer]
        # small_partition = df_examine[pointer:end_pointer]*taper_windows
        welch_small_partition = signal.welch(small_partition, window=window)
        peaks_small_partition = signal.find_peaks(welch_small_partition[1],
                                                  threshold=np.mean(welch_small_partition[1]))
        if len(peaks_small_partition[0]) > len(peaks_full[0]) * peak_threshold_ratio:
            remove_start_indices.append(pointer)
            remove_end_indices.append(end_pointer)

        pointer = pointer + int(window_size * overlap_rate)

    start_trim_by_freq, end_trim_by_freq = concate_remove_index(remove_start_indices, remove_end_indices,
                                                                remove_sliding_window)
    start_milestone_by_freq,end_milestone_by_freq = \
        cut_milestone_to_keep_milestone(start_trim_by_freq, end_trim_by_freq,len(df_examine))
    # if 0 not in np.array(start_trim_by_freq):
    #     start_milestone_by_freq = np.hstack((0, np.array(end_trim_by_freq) + 1))
    #     end_milestone_by_freq = np.hstack((np.array(start_trim_by_freq) - 1, len(df) - 1))
    # else:
    #     start_milestone_by_freq = np.array(end_trim_by_freq) + 1
    #     end_milestone_by_freq = np.hstack((np.array(start_trim_by_freq)[1:] - 1, len(df) - 1))
    return start_milestone_by_freq,end_milestone_by_freq
