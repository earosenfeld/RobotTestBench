"""
Reusable UI components for the RobotTestBench dashboard.
"""

from dash import html, dcc
import plotly.graph_objs as go

def create_control_panel():
    """Create the control panel component."""
    return html.Div([
        html.H3("Control Panel"),
        html.Div([
            html.Label("Control Mode:"),
            dcc.Dropdown(
                id='control-mode',
                options=[
                    {'label': 'Position', 'value': 'position'},
                    {'label': 'Velocity', 'value': 'velocity'},
                    {'label': 'Torque', 'value': 'torque'}
                ],
                value='position'
            ),
            html.Label("Setpoint:"),
            dcc.Input(id='setpoint', type='number', value=0),
            html.Button('Start Test', id='start-test', n_clicks=0),
            html.Button('Stop Test', id='stop-test', n_clicks=0),
        ], style={'padding': '10px'})
    ], style={'width': '30%', 'float': 'left'})

def create_plots_panel():
    """Create the real-time plots panel."""
    return html.Div([
        html.H3("Real-time Data"),
        dcc.Graph(id='position-plot'),
        dcc.Graph(id='velocity-plot'),
        dcc.Graph(id='torque-plot'),
        dcc.Interval(
            id='interval-component',
            interval=100,  # in milliseconds
            n_intervals=0
        )
    ], style={'width': '70%', 'float': 'right'})

def create_config_panel():
    """Create the test configuration panel."""
    return html.Div([
        html.H3("Test Configuration"),
        html.Div([
            html.Label("Motor Type:"),
            dcc.Input(id='motor-type', type='text', value='Simulated'),
            html.Label("Test Type:"),
            dcc.Input(id='test-type', type='text', value='Step Response'),
            html.Label("Description:"),
            dcc.Textarea(id='test-description', value=''),
        ], style={'padding': '10px'})
    ], style={'width': '30%', 'float': 'left', 'margin-top': '20px'})

def create_plot_template(title: str, yaxis_title: str) -> go.Figure:
    """Create a basic plot template."""
    return {
        'data': [{
            'x': [],
            'y': [],
            'type': 'scatter',
            'mode': 'lines',
            'name': title
        }],
        'layout': {
            'title': title,
            'xaxis': {'title': 'Time (s)'},
            'yaxis': {'title': yaxis_title}
        }
    } 