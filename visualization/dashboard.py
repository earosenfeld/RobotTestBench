"""
Real-time visualization dashboard for RobotTestBench.
Uses Plotly Dash for interactive plotting.
"""

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
from datetime import datetime
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any

# Initialize the Dash app
app = dash.Dash(__name__)

def create_layout():
    """Create the dashboard layout."""
    return html.Div([
        html.H1("RobotTestBench Dashboard"),
        
        # Control Panel
        html.Div([
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
        ], style={'width': '30%', 'float': 'left'}),
        
        # Real-time Plots
        html.Div([
            html.H3("Real-time Data"),
            dcc.Graph(id='position-plot'),
            dcc.Graph(id='velocity-plot'),
            dcc.Graph(id='torque-plot'),
            dcc.Interval(
                id='interval-component',
                interval=100,  # in milliseconds
                n_intervals=0
            )
        ], style={'width': '70%', 'float': 'right'}),
        
        # Test Configuration
        html.Div([
            html.H3("Test Configuration"),
            html.Div([
                html.Label("Motor Type:"),
                dcc.Input(id='motor-type', type='text', value='Simulated'),
                html.Label("Test Type:"),
                dcc.Input(id='test-type', type='text', value='Step Response'),
                html.Label("Description:"),
                dcc.Textarea(id='test-description', value=''),
            ], style={'padding': '10px'})
        ], style={'width': '30%', 'float': 'left', 'margin-top': '20px'}),
        
        # Hidden div for storing test data
        html.Div(id='test-data', style={'display': 'none'})
    ])

def create_plot(title: str, yaxis_title: str) -> go.Figure:
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

# Set up the layout
app.layout = create_layout()

# Callback for updating plots
@app.callback(
    [Output('position-plot', 'figure'),
     Output('velocity-plot', 'figure'),
     Output('torque-plot', 'figure')],
    [Input('interval-component', 'n_intervals')],
    [State('test-data', 'children')]
)
def update_plots(n_intervals, test_data):
    """Update the real-time plots."""
    if not test_data:
        return (
            create_plot('Position', 'Position (rad)'),
            create_plot('Velocity', 'Velocity (rad/s)'),
            create_plot('Torque', 'Torque (N⋅m)')
        )
    
    data = json.loads(test_data)
    df = pd.DataFrame(data)
    
    position_fig = {
        'data': [{
            'x': df['elapsed_time'],
            'y': df['position'],
            'type': 'scatter',
            'mode': 'lines',
            'name': 'Position'
        }],
        'layout': {
            'title': 'Position',
            'xaxis': {'title': 'Time (s)'},
            'yaxis': {'title': 'Position (rad)'}
        }
    }
    
    velocity_fig = {
        'data': [{
            'x': df['elapsed_time'],
            'y': df['velocity'],
            'type': 'scatter',
            'mode': 'lines',
            'name': 'Velocity'
        }],
        'layout': {
            'title': 'Velocity',
            'xaxis': {'title': 'Time (s)'},
            'yaxis': {'title': 'Velocity (rad/s)'}
        }
    }
    
    torque_fig = {
        'data': [{
            'x': df['elapsed_time'],
            'y': df['torque'],
            'type': 'scatter',
            'mode': 'lines',
            'name': 'Torque'
        }],
        'layout': {
            'title': 'Torque',
            'xaxis': {'title': 'Time (s)'},
            'yaxis': {'title': 'Torque (N⋅m)'}
        }
    }
    
    return position_fig, velocity_fig, torque_fig

def launch_dashboard():
    """Launch the dashboard application."""
    app.run_server(debug=True, port=8050)

if __name__ == '__main__':
    launch_dashboard() 