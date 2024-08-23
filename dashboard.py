import dash
from dash import dcc, html
import plotly.graph_objs as go
import json
import os
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import glob

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MORPH])

# Function to load machine data from JSON files
def load_machine_data():
    data = {}
    for machine in ['machine1', 'machine2', 'machine3']:
        with open(f'{machine}.json', 'r') as f:
            data[machine] = json.load(f)
    return data

# Function to load process data from JSON files
def load_process_data():
    data = {}
    process_files = glob.glob('process*.json')
    for process_file in process_files:
        with open(process_file, 'r') as f:
            process_name = os.path.splitext(os.path.basename(process_file))[0]
            data[process_name] = json.load(f)
    return data

# Create dropdown options for machine metrics
machine_dropdown_options = [
    {'label': 'CPU', 'value': 'CPU'},
    {'label': 'Memory', 'value': 'Memory'},
    {'label': 'Temperature', 'value': 'Temp'},
    {'label': 'Power Consumption', 'value': 'Power'}
]

# Create dropdown options for process metrics
process_dropdown_options = [
    {'label': 'Training Loss', 'value': 'training_loss'},
    {'label': 'Time per Epoch', 'value': 'time_per_epoch'},
    {'label': 'F1 Score', 'value': 'f1_score'}
]

# Create radio button options for machine view selection
machine_view_options = [
    {'label': 'Combined View', 'value': 'combined'},
    {'label': 'Machine 1', 'value': 'machine1'},
    {'label': 'Machine 2', 'value': 'machine2'},
    {'label': 'Machine 3', 'value': 'machine3'}
]

# Layout of the app
app.layout = dbc.Container([
    dcc.Tabs([
        dcc.Tab(label='Machine Monitoring', children=[
            html.H1('Machine Monitoring Dashboard', className='text-white'),
            dbc.Row([
                dbc.Col([
                    dcc.RadioItems(
                        id='view-radio',
                        options=machine_view_options,
                        value='combined',
                    ),
                ], width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Dropdown(
                        id='metric-dropdown',
                        options=machine_dropdown_options,
                        value='CPU',
                    ),
                ], width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='metric-graph'),
                ], width=12)
            ]),
            dcc.Interval(
                id='interval-component',
                interval=1000,  # in milliseconds (e.g., 60 seconds)
                n_intervals=0
            )
        ]),
        dcc.Tab(label='Processes View', children=[
            html.H1('Processes Monitoring Dashboard', className='text-white'),
            dbc.Row([
                dbc.Col([
                    dcc.Dropdown(
                        id='process-dropdown',
                        options=[],  # Will be populated dynamically
                        value=None,
                    ),
                ], width=6),
                dbc.Col([
                    dcc.Dropdown(
                        id='process-metric-dropdown',
                        options=process_dropdown_options,
                        value='training_loss',
                    ),
                ], width=6)
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='process-metric-graph'),
                ], width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Progress(id='process-progress-bar', striped=True, animated=True, style={"height": "30px"}),
                ], width=12)
            ]),
            dcc.Interval(
                id='process-interval-component',
                interval=1000,  # in milliseconds (e.g., 60 seconds)
                n_intervals=0
            )
        ])
    ])
], fluid=True)

# Callback to update machine graph based on selected metric and view
@app.callback(
    Output('metric-graph', 'figure'),
    [Input('metric-dropdown', 'value'),
     Input('view-radio', 'value'),
     Input('interval-component', 'n_intervals')]
)
def update_machine_graph(selected_metric, selected_view, n_intervals):
    data = load_machine_data()  # Reload data at each interval
    traces = []

    if selected_view == 'combined':
        for machine, machine_data in data.items():
            traces.append(go.Scatter(
                x=list(range(len(machine_data[selected_metric]))),
                y=machine_data[selected_metric],
                mode='lines',
                name=machine
            ))
    else:
        machine_data = data[selected_view]
        traces.append(go.Scatter(
            x=list(range(len(machine_data[selected_metric]))),
            y=machine_data[selected_metric],
            mode='lines',
            name=selected_view
        ))

    return {
        'data': traces,
        'layout': go.Layout(
            title=f'{selected_metric} over Time',
            xaxis={'title': 'Time'},
            yaxis={'title': selected_metric}
        )
    }

# Callback to update process dropdown options
@app.callback(
    Output('process-dropdown', 'options'),
    [Input('process-interval-component', 'n_intervals')]
)
def update_process_dropdown(n_intervals):
    process_files = glob.glob('process*.json')
    options = [{'label': os.path.splitext(os.path.basename(f))[0], 'value': os.path.splitext(os.path.basename(f))[0]} for f in process_files]
    return options

# Callback to update process graph and progress bar based on selected metric and process
@app.callback(
    [Output('process-metric-graph', 'figure'),
     Output('process-progress-bar', 'value')],
    [Input('process-metric-dropdown', 'value'),
     Input('process-dropdown', 'value'),
     Input('process-interval-component', 'n_intervals')]
)
def update_process_graph(selected_metric, selected_process, n_intervals):
    data = load_process_data()  # Reload data at each interval
    traces = []
    progress = 0

    if selected_process:
        process_data = data[selected_process]
        traces.append(go.Scatter(
            x=list(range(len(process_data[selected_metric]))),
            y=process_data[selected_metric],
            mode='lines',
            name=selected_process
        ))
        progress = process_data['progress'][-1] if process_data['progress'] else 0

    return {
        'data': traces,
        'layout': go.Layout(
            title=f'{selected_metric} over Time',
            xaxis={'title': 'Time'},
            yaxis={'title': selected_metric}
        )
    }, progress

if __name__ == '__main__':
    app.run_server()