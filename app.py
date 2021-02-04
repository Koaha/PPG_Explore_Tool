# https://towardsdatascience.com/creating-an-interactive-data-app-using-plotlys-dash-356428b4699c

import plotly.graph_objs as go
import plotly.express as px
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import numpy as np
import plotly.graph_objects as go
import sys
import os
sys.path.append(os.path.join(os.getcwd(),".."))
# print(os.path.join(os.getcwd(),"..","utilties"))
import pandas as pd

# from utilities.trim_utilities import is_start,is_end,parse_data
#try:
#    from .trim_utilities import *
#except Exception as e:
#    from trim_utilities import *

#DATA_PATH = os.path.join(os.getcwd(),"..","data")
MIN_CYCLE = 5
MIN_MINUTES = 5
SAMPLE_RATE = 100
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

global df,df_cut
df = pd.DataFrame()
df_cut = []

global start_milestone,end_milestone
start_milestone = []
end_milestone = []

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

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

colors = {
    "graphBackground": "#F5F5F5",
    "background": "#ffffff",
    "text": "#000000"
}

app.layout = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '50px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '2px',
            'textAlign': 'center',
            'margin': '2px'
        },
        # Allow multiple files to be uploaded
        multiple=False
    ),
    dcc.Graph(id='Mygraph'),
    # dcc.Graph(id='Trimgraph'),
    html.Div([
            html.Button('Trim Data', id='trim_button', n_clicks=0)
            ]),
    html.Details([
        html.Div(dcc.Input(id='input-window-size', type='text')),
        html.P('window-size (Default= Min cycle * samle rate (5 * 100)'),

        html.Div(dcc.Input(id='input-peak-threshold', type='text')),
        html.P('peak-ratio-threshold (Default = 1.8 - Increase if the window-size increases)'),

        html.Div(dcc.Input(id='input-remove-sliding-window', type='text')),
        html.P('sliding window to concatenate the removal milestone (Default = 0)')
       
    ]),
    html.Div([
        html.Button('Trim By Frequency', id='trim_freq_button', n_clicks=0),
        html.Button('Save Data Split', id='save_button', n_clicks=0),
        html.Div(id='save_message')
        ]),
    html.Div(id='output-data-upload')
])

@app.callback(
    Output('save_message','children'),
    [
        Input('save_button','n_clicks'),
        Input('upload-data', 'contents'),
        Input('upload-data', 'filename')
    ]
)
def save_split_data(btn_save,content,filename):
    if btn_save:
        fname = filename.split(".")[0]
        inc = 1
        for df_small_chunk in df_cut:
            #fname_part = "-".join((fname,str(inc)))+".csv"
            #full_fn = os.path.join(DATA_PATH,fname_part)
            #df_small_chunk.to_csv(full_fn)
            inc+=1
        return "Saved successfully!"


@app.callback(
    Output('Mygraph', 'figure'),
    [
        Input('upload-data', 'contents'),
        Input('upload-data', 'filename'),
        Input('trim_button', 'n_clicks'),
        Input('trim_freq_button', 'n_clicks'),
        dash.dependencies.State('input-window-size', 'value'),
        dash.dependencies.State('input-peak-threshold', 'value'),
        dash.dependencies.State('input-remove-sliding-window', 'value'),
    ])
def update_graph(contents, filename, btn_trim, btn_trim_freq,
                 input_window_size,input_peak_threshold,input_remove_sliding_window):
    global df,start_milestone,end_milestone,df_cut
    fig = {
        'layout': go.Layout(
            plot_bgcolor=colors["graphBackground"],
            paper_bgcolor=colors["graphBackground"])
    }

    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    if 'trim_button' in changed_id:
        print("TRIM BUTTON CLICK")
        start_milestone, end_milestone = trim_missing_signal(df)
        start_milestone,end_milestone = remove_short_length(start_milestone,end_milestone,MIN_CYCLE*SAMPLE_RATE)

        print(start_milestone)
        print(end_milestone)
         # UPDATE FIGURE
        fig = go.Figure()
        for start, end in zip(start_milestone, end_milestone):
                fig.add_traces(
                    go.Scatter(
                        x=df["PLETH"][int(start):int(end)].index,
                        y=df["PLETH"][int(start):int(end)],
                        mode="lines"
                    ))
        return fig
    elif 'trim_freq_button' in changed_id:
        print("TRIM BUTTON FREQ CLICK")
        fig = go.Figure()
        if len(start_milestone) == 0 or len(end_milestone)==0:
            return  fig
        df_cut = []

        try:
            window_size = int(input_window_size)
        except Exception as error:
            window_size = SAMPLE_RATE * MIN_CYCLE
        try:
            peak_threshold = float(input_peak_threshold)
        except Exception as error:
            peak_threshold = None
        try:
            remove_sliding_window = int(input_remove_sliding_window)
        except Exception as error:
            remove_sliding_window = None


        for start_idx,end_idx in zip(start_milestone,end_milestone):
            df_examine = df.iloc[int(start_idx):int(end_idx)]
            start_milestone_by_freq, end_milestone_by_freq = \
                trim_by_frequency_partition(df_examine["PLETH"],
                                            start_milestone,
                                            end_milestone,
                                            window_size=window_size,
                                            peak_threshold_ratio = peak_threshold,
                                            remove_sliding_window = remove_sliding_window)

            start_milestone_by_freq, end_milestone_by_freq = \
                remove_short_length(start_milestone_by_freq, end_milestone_by_freq,
                                                                 MIN_MINUTES * 60 * SAMPLE_RATE)

            # UPDATE FIGURE
            for start, end in zip(start_milestone_by_freq, end_milestone_by_freq):
                fig.add_traces(
                    go.Scatter(
                        x=df_examine["PLETH"].iloc[int(start):int(end)].index,
                        y=df_examine["PLETH"].iloc[int(start):int(end)],
                        mode="lines"
                    ))
                df_cut.append(df_examine.iloc[int(start):int(end)])
            
        return fig

    elif 'upload-data' in changed_id:
        print("GET CONTENT")
        df = parse_data(contents, filename)
        fig = go.Figure()
        fig.add_traces(go.Scatter(x=df["PLETH"].index, y=df["PLETH"], mode="lines"))

    return fig



if __name__ == '__main__':
    app.run_server(debug=True)
    # app.run_server(debug=True, host='0.0.0.0', port=8000)
