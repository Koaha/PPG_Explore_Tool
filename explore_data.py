# https://towardsdatascience.com/creating-an-interactive-data-app-using-plotlys-dash-356428b4699c

import plotly.graph_objs as go
import cufflinks as cf
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
try:
    from .trim_utilities import *
except Exception as e:
    from trim_utilities import *

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
        # html.Div
        #     .Textarea(id='container-button-basic',
        #          children='Enter a value to trim by frequency')
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
            # print(df_start_idx)
            # print(df_end_idx)
            # df_start_idx = np.hstack((df_start_idx,
            #                          [df_examine.index[df_examine.iloc[int(milestone)]]
            #                           for milestone in start_milestone_by_freq]))
            # df_end_idx = np.hstack((df_end_idx,
            #                        [df_examine.index[df_examine.iloc[int(milestone)]]
            #                           for milestone in end_milestone_by_freq]))
        return fig

    elif 'upload-data' in changed_id:
        print("GET CONTENT")
        df = parse_data(contents, filename)
        fig = go.Figure()
        fig.add_traces(go.Scatter(x=df["PLETH"].index, y=df["PLETH"], mode="lines"))

    return fig


# @app.callback(Output('output-data-upload', 'children'),
#             [
#                 Input('upload-data', 'contents'),
#                 Input('upload-data', 'filename')
#             ])
# def update_table(contents, filename):
#     table = html.Div()
#
#     if contents:
#         contents = contents[0]
#         filename = filename[0]
#         df = parse_data(contents, filename)
#
#         table = html.Div([
#             html.H5(filename),
#             dash_table.DataTable(
#                 data=df.to_dict('rows'),
#                 columns=[{'name': i, 'id': i} for i in df.columns]
#             ),
#             html.Hr(),
#             html.Div('Raw Content'),
#             html.Pre(contents[0:200] + '...', style={
#                 'whiteSpace': 'pre-wrap',
#                 'wordBreak': 'break-all'
#             })
#         ])
#
#     return table

if __name__ == '__main__':
    app.run_server(debug=True)
    # app.run_server(debug=True, host='0.0.0.0', port=8000)
