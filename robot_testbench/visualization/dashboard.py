"""
Real-time visualization dashboard for RobotTestBench.
Uses Plotly Dash for interactive plotting.
"""

import dash
from dash import html
from .components import create_control_panel, create_plots_panel, create_config_panel
from .callbacks import register_callbacks

# Initialize the Dash app
app = dash.Dash(__name__)

def create_layout():
    """Create the dashboard layout."""
    return html.Div([
        html.H1("RobotTestBench Dashboard"),
        
        # Control Panel
        create_control_panel(),
        
        # Real-time Plots
        create_plots_panel(),
        
        # Test Configuration
        create_config_panel(),
        
        # Hidden div for storing test data
        html.Div(id='test-data', style={'display': 'none'})
    ])

# Set up the layout
app.layout = create_layout()

# Register callbacks
register_callbacks(app)

def launch_dashboard():
    """Launch the dashboard application."""
    app.run(debug=True, port=8050)

if __name__ == '__main__':
    launch_dashboard() 