"""
Callback functions for the RobotTestBench dashboard.
"""

from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
from visualization.components import create_plot_template
from visualization.data_manager import TestDataManager

# Initialize data manager
data_manager = TestDataManager()

def register_callbacks(app):
    """Register all dashboard callbacks."""
    
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
                create_plot_template('Position', 'Position (rad)'),
                create_plot_template('Velocity', 'Velocity (rad/s)'),
                create_plot_template('Torque', 'Torque (N⋅m)')
            )
        
        data = data_manager.get_dataframe()
        
        position_fig = {
            'data': [{
                'x': data['elapsed_time'],
                'y': data['position'],
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
                'x': data['elapsed_time'],
                'y': data['velocity'],
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
                'x': data['elapsed_time'],
                'y': data['torque'],
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
    
    @app.callback(
        Output('test-data', 'children'),
        [Input('start-test', 'n_clicks'),
         Input('stop-test', 'n_clicks')],
        [State('control-mode', 'value'),
         State('setpoint', 'value')]
    )
    def handle_test_control(start_clicks, stop_clicks, control_mode, setpoint):
        """Handle test start/stop events."""
        if start_clicks > 0:
            data_manager.start_test()
        elif stop_clicks > 0:
            data_manager.stop_test()
            # Save test data
            data_manager.save_test_data('test_data.csv')
            
        return data_manager.get_current_data() 