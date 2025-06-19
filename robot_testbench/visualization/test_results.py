"""
Test results visualization module for RobotTestBench.
Creates professional test result dashboards using Plotly and Dash.
"""

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from ..analytics.test_analyzer import Analyzer, PerformanceMetrics

class ResultsDashboard:
    """Creates and manages the test results dashboard."""
    
    def __init__(self, data_dir: str = "data/logs"):
        self.data_dir = Path(data_dir)
        logger.info(f"Initializing dashboard with data directory: {self.data_dir}")
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.register_callbacks()
        
    def setup_layout(self):
        """Setup the dashboard layout."""
        self.app.layout = html.Div([
            html.H1("RobotTestBench - Test Results Dashboard", 
                   style={'textAlign': 'center', 'margin': '20px'}),
            
            # Test Selection
            html.Div([
                html.H3("Test Selection"),
                html.Div([
                    html.Div([
                        html.H4("Single Test View"),
                        dcc.Dropdown(
                            id='test-selector',
                            options=self._get_test_options(),
                            placeholder="Select a test..."
                        )
                    ], style={'width': '50%', 'display': 'inline-block'}),
                    html.Div([
                        html.H4("Test Comparison"),
                        dcc.Dropdown(
                            id='test-comparison-selector',
                            options=self._get_test_options(),
                            placeholder="Select tests to compare...",
                            multi=True
                        )
                    ], style={'width': '50%', 'display': 'inline-block'})
                ])
            ], style={'width': '90%', 'margin': '20px auto'}),
            
            # Hardware Configuration
            html.Div([
                html.H3("Hardware Configuration"),
                html.Div([
                    html.Div([
                        html.H4("Motor Specifications"),
                        html.Div(id='motor-specs', style={'margin': '20px'})
                    ], style={'width': '50%', 'display': 'inline-block'}),
                    html.Div([
                        html.H4("Sensor Specifications"),
                        html.Div(id='sensor-specs', style={'margin': '20px'})
                    ], style={'width': '50%', 'display': 'inline-block'})
                ])
            ]),
            
            # System Configuration
            html.Div([
                html.H3("System Configuration"),
                html.Div([
                    html.Div([
                        html.H4("Motor Parameters"),
                        html.Div(id='motor-params', style={'margin': '20px'})
                    ], style={'width': '50%', 'display': 'inline-block'}),
                    html.Div([
                        html.H4("PID Gains"),
                        html.Div(id='pid-gains', style={'margin': '20px'})
                    ], style={'width': '50%', 'display': 'inline-block'})
                ]),
                html.Div([
                    html.H4("Sensor Configuration"),
                    html.Div(id='sensor-config', style={'margin': '20px'})
                ])
            ]),
            
            # Test Summary
            html.Div([
                html.H3("Test Summary"),
                html.Div(id='test-summary', style={'margin': '20px'})
            ]),
            
            # Performance Metrics
            html.Div([
                html.H3("Performance Metrics"),
                html.Div(id='performance-metrics', style={'margin': '20px'})
            ]),
            
            # Test Comparison Section
            html.Div([
                html.H3("Test Comparison"),
                html.Div(id='test-comparison', style={'margin': '20px'})
            ]),
            
            # Time Series Plots
            html.Div([
                html.H3("Test Data"),
                dcc.Graph(id='position-plot'),
                dcc.Graph(id='velocity-plot'),
                dcc.Graph(id='torque-plot'),
                dcc.Graph(id='current-plot')
            ]),
            
            # Sensor Data Section
            html.Div([
                html.H3("Sensor Data Analysis"),
                html.Div([
                    html.Div([
                        html.H4("Raw vs Filtered Data"),
                        dcc.Graph(id='sensor-comparison-plot')
                    ], style={'width': '100%'}),
                    html.Div([
                        html.H4("Sensor Noise Analysis"),
                        dcc.Graph(id='sensor-noise-plot')
                    ], style={'width': '100%'})
                ])
            ]),
            
            # Thermal Analysis Section
            html.Div([
                html.H3("Thermal Analysis"),
                html.Div([
                    html.Div([
                        html.H4("Temperature Profile"),
                        dcc.Graph(id='temperature-plot')
                    ], style={'width': '100%'}),
                    html.Div([
                        html.H4("Thermal Model"),
                        dcc.Graph(id='thermal-model-plot')
                    ], style={'width': '100%'})
                ])
            ]),
            
            # Hidden div for storing test data
            html.Div(id='test-data', style={'display': 'none'})
        ])
        
    def _get_test_options(self) -> List[Dict[str, str]]:
        """Get list of available tests for the dropdown."""
        options = []
        for test_dir in self.data_dir.glob("*"):
            if test_dir.is_dir():
                options.append({
                    'label': test_dir.name,
                    'value': test_dir.name
                })
        logger.info(f"Found test options: {options}")
        return options
        
    def _create_empty_figure(self, title: str) -> go.Figure:
        """Create an empty figure with a title."""
        fig = go.Figure()
        fig.update_layout(
            title=title,
            xaxis_title='Time (s)',
            yaxis_title='Value',
            showlegend=True
        )
        return fig
        
    def register_callbacks(self):
        """Register dashboard callbacks."""
        
        @self.app.callback(
            [Output('test-summary', 'children'),
             Output('performance-metrics', 'children'),
             Output('motor-params', 'children'),
             Output('pid-gains', 'children'),
             Output('sensor-config', 'children'),
             Output('motor-specs', 'children'),
             Output('sensor-specs', 'children'),
             Output('position-plot', 'figure'),
             Output('velocity-plot', 'figure'),
             Output('torque-plot', 'figure'),
             Output('current-plot', 'figure'),
             Output('sensor-comparison-plot', 'figure'),
             Output('sensor-noise-plot', 'figure'),
             Output('temperature-plot', 'figure'),
             Output('thermal-model-plot', 'figure')],
            [Input('test-selector', 'value')]
        )
        def update_dashboard(test_name):
            print(f"\n=== Dashboard Update ===")
            print(f"Selected test: {test_name}")
            
            if not test_name:
                print("No test selected, returning empty state")
                empty_div = html.Div("Select a test to view results")
                empty_fig = self._create_empty_figure("")
                return [empty_div] * 7 + [empty_fig] * 8
                
            # Load test data
            print(f"\nLoading test data for: {test_name}")
            test_data = self._load_test_data(test_name)
            if not test_data:
                print("Failed to load test data")
                error_div = html.Div("Error loading test data")
                empty_fig = self._create_empty_figure("")
                return [error_div] * 7 + [empty_fig] * 8
                
            print("\nTest data loaded successfully")
            print(f"Metadata keys: {test_data['metadata'].keys()}")
            print(f"Data columns: {test_data['data'].columns.tolist()}")
            print(f"Number of data points: {len(test_data['data'])}")
            
            # Check if data is empty (all zeros)
            if test_data.get('is_empty', False):
                return [
                    html.Div([
                        html.H4("Test Summary"),
                        html.P(f"Test Name: {test_data['metadata']['test_name']}"),
                        html.P(f"Date: {test_data['metadata']['timestamp']}"),
                        html.P(f"Control Mode: {test_data['metadata']['control_mode']}"),
                        html.P(f"Setpoint: {test_data['metadata']['setpoint']:.2f}"),
                        html.P(f"Duration: {test_data['metadata']['duration']:.2f} s"),
                        html.P("Note: This test contains no meaningful data (all values are zero)")
                    ]),
                    html.Div("No performance metrics available for empty test data"),
                    self._create_motor_params_table(test_data['metadata']),
                    self._create_pid_gains_table(test_data['metadata']),
                    self._create_sensor_config_table(test_data['metadata']),
                    self._create_motor_specs_table(test_data['metadata']),
                    self._create_sensor_specs_table(test_data['metadata']),
                    self._create_empty_figure("Position vs Time"),
                    self._create_empty_figure("Velocity vs Time"),
                    self._create_empty_figure("Torque vs Time"),
                    self._create_empty_figure("Current vs Time"),
                    self._create_empty_figure("Sensor Data Comparison"),
                    self._create_empty_figure("Sensor Noise Analysis"),
                    self._create_empty_figure("Temperature Profile"),
                    self._create_empty_figure("Thermal Model")
                ]
            
            # Create summary statistics
            print("\nCreating summary statistics")
            summary = self._create_summary(test_data)
            
            # Calculate performance metrics
            print("\nCalculating performance metrics")
            metrics = self._calculate_metrics(test_data)
            
            # Create configuration tables
            motor_params = self._create_motor_params_table(test_data['metadata'])
            pid_gains = self._create_pid_gains_table(test_data['metadata'])
            sensor_config = self._create_sensor_config_table(test_data['metadata'])
            motor_specs = self._create_motor_specs_table(test_data['metadata'])
            sensor_specs = self._create_sensor_specs_table(test_data['metadata'])
            
            # Create plots
            print("\nCreating plots")
            position_fig = self._create_position_plot(test_data)
            velocity_fig = self._create_velocity_plot(test_data)
            torque_fig = self._create_torque_plot(test_data)
            current_fig = self._create_current_plot(test_data)
            sensor_comparison_fig = self._create_sensor_comparison_plot(test_data)
            sensor_noise_fig = self._create_sensor_noise_plot(test_data)
            temperature_fig = self._create_temperature_plot(test_data)
            thermal_model_fig = self._create_thermal_model_plot(test_data)
            
            print("=== Dashboard Update Complete ===\n")
            return (summary, metrics, motor_params, pid_gains, sensor_config,
                   motor_specs, sensor_specs,
                   position_fig, velocity_fig, torque_fig, current_fig,
                   sensor_comparison_fig, sensor_noise_fig,
                   temperature_fig, thermal_model_fig)
            
        @self.app.callback(
            Output('test-comparison', 'children'),
            [Input('test-comparison-selector', 'value')]
        )
        def update_test_comparison(test_names):
            if not test_names or len(test_names) < 2:
                return html.Div("Select at least two tests to compare")
                
            # Load test data for all selected tests
            test_data_list = []
            for test_name in test_names:
                test_data = self._load_test_data(test_name)
                if test_data and not test_data.get('is_empty', False):
                    test_data_list.append((test_name, test_data))
                    
            if len(test_data_list) < 2:
                return html.Div("Need at least two valid tests to compare")
                
            # Create comparison tables and plots
            return html.Div([
                # Configuration Comparison
                html.Div([
                    html.H4("Configuration Comparison"),
                    self._create_config_comparison_table(test_data_list)
                ]),
                
                # Performance Metrics Comparison
                html.Div([
                    html.H4("Performance Metrics Comparison"),
                    self._create_metrics_comparison_table(test_data_list)
                ]),
                
                # A/B Testing Visualization
                html.Div([
                    html.H4("A/B Testing Visualization"),
                    dcc.Graph(
                        figure=self._create_ab_testing_plot(test_data_list)
                    )
                ])
            ])
            
    def _load_test_data(self, test_name: str) -> Optional[Dict]:
        """Load test data from files."""
        test_dir = self.data_dir / test_name
        print(f"\nLoading test data from: {test_dir}")
        
        if not test_dir.exists():
            print(f"Error: Test directory does not exist: {test_dir}")
            return None
            
        try:
            # Load metadata
            metadata_path = test_dir / "metadata.json"
            print(f"Loading metadata from: {metadata_path}")
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            print(f"Metadata loaded: {metadata}")
                
            # Load data
            data_path = test_dir / "data.csv"
            print(f"Loading data from: {data_path}")
            data = pd.read_csv(data_path)
            print(f"Data loaded: {len(data)} rows")
            print(f"Data columns: {data.columns.tolist()}")
            print(f"First few rows:\n{data.head()}")
            print(f"Data types:\n{data.dtypes}")
            print(f"Data info:\n{data.info()}")
            
            # Verify data is not empty
            if len(data) == 0:
                print("Error: Data is empty")
                return None
                
            # Verify required columns exist
            required_columns = ['elapsed_time', 'filtered_position', 'filtered_velocity', 'torque', 'filtered_current']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                print(f"Error: Missing required columns: {missing_columns}")
                return None
                
            # Verify data is numeric
            for col in required_columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    print(f"Error: Column {col} is not numeric")
                    return None
                    
            # Check if data has meaningful values (not all zeros)
            meaningful_data = False
            for col in ['filtered_position', 'filtered_velocity', 'torque', 'filtered_current']:
                if data[col].abs().max() > 1e-6:  # Check if any value is significantly different from zero
                    meaningful_data = True
                    break
                    
            if not meaningful_data:
                print("Warning: Data contains only zeros or very small values")
                return {
                    "metadata": metadata,
                    "data": data,
                    "is_empty": True
                }
            
            return {
                "metadata": metadata,
                "data": data,
                "is_empty": False
            }
        except Exception as e:
            print(f"Error loading test data: {str(e)}")
            import traceback
            print(f"Traceback:\n{traceback.format_exc()}")
            return None
        
    def _create_summary(self, test_data: Dict) -> html.Div:
        """Create summary statistics section."""
        metadata = test_data["metadata"]
        print("\nCreating summary with metadata:", metadata)
        
        return html.Div([
            html.Table([
                html.Tr([
                    html.Td("Test Name:"),
                    html.Td(metadata["test_name"])
                ]),
                html.Tr([
                    html.Td("Date:"),
                    html.Td(metadata["timestamp"])
                ]),
                html.Tr([
                    html.Td("Control Mode:"),
                    html.Td(metadata["control_mode"])
                ]),
                html.Tr([
                    html.Td("Setpoint:"),
                    html.Td(f"{metadata['setpoint']:.2f}")
                ]),
                html.Tr([
                    html.Td("Duration:"),
                    html.Td(f"{metadata['duration']:.2f} s")
                ])
            ], style={'margin': '20px auto'})
        ])
        
    def _calculate_metrics(self, test_data: Dict) -> html.Div:
        """Calculate and display performance metrics."""
        data = test_data["data"]
        setpoint = test_data["metadata"]["setpoint"]
        print(f"\nCalculating metrics with setpoint: {setpoint}")
        
        analyzer = Analyzer(data, setpoint)
        metrics = analyzer.compute_metrics()
        print(f"Calculated metrics: {metrics}")
        
        return html.Div([
            html.Table([
                html.Tr([
                    html.Td("Rise Time:"),
                    html.Td(f"{metrics.rise_time:.3f} s")
                ]),
                html.Tr([
                    html.Td("Settling Time:"),
                    html.Td(f"{metrics.settling_time:.3f} s")
                ]),
                html.Tr([
                    html.Td("Overshoot:"),
                    html.Td(f"{metrics.overshoot:.1f}%")
                ]),
                html.Tr([
                    html.Td("Steady State Error:"),
                    html.Td(f"{metrics.steady_state_error:.3f}")
                ]),
                html.Tr([
                    html.Td("RMS Error:"),
                    html.Td(f"{metrics.rms_error:.3f}")
                ]),
                html.Tr([
                    html.Td("Peak Torque:"),
                    html.Td(f"{metrics.peak_torque:.2f} N⋅m")
                ]),
                html.Tr([
                    html.Td("Peak Velocity:"),
                    html.Td(f"{metrics.peak_velocity:.2f} rad/s")
                ])
            ], style={'margin': '20px auto'})
        ])
        
    def _create_position_plot(self, test_data: Dict) -> go.Figure:
        """Create a plot showing position data."""
        data = test_data['data']
        metadata = test_data['metadata']
        setpoint = metadata['setpoint']
        
        fig = make_subplots(rows=2, cols=1,
                           subplot_titles=('Position vs Time', 'Position Error Analysis'),
                           vertical_spacing=0.1)
        
        # Main position plot
        fig.add_trace(
            go.Scatter(x=data['elapsed_time'], y=data['filtered_position'],
                      name='Actual Position', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=data['elapsed_time'], y=[setpoint] * len(data),
                      name='Setpoint', line=dict(color='red', dash='dash')),
            row=1, col=1
        )
        
        # Position error analysis
        position_error = data['filtered_position'] - setpoint
        fig.add_trace(
            go.Scatter(x=data['elapsed_time'], y=position_error,
                      name='Position Error', line=dict(color='orange')),
            row=2, col=1
        )
        
        # Add error bounds
        error_std = position_error.std()
        fig.add_trace(
            go.Scatter(x=data['elapsed_time'], y=[2*error_std]*len(data),
                      name='+2σ Bound', line=dict(color='gray', dash='dot')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=data['elapsed_time'], y=[-2*error_std]*len(data),
                      name='-2σ Bound', line=dict(color='gray', dash='dot')),
            row=2, col=1
        )
        
        fig.update_layout(
            title='Position Control Analysis',
            height=800,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
        
    def _create_velocity_plot(self, test_data: Dict) -> go.Figure:
        """Create a plot showing velocity data."""
        data = test_data['data']
        metadata = test_data['metadata']
        velocity_setpoint = metadata.get('velocity_setpoint', 0)
        fig = make_subplots(rows=3, cols=1,
                           subplot_titles=('Velocity vs Time', 'Velocity Error Analysis', 'Jerk Analysis'),
                           vertical_spacing=0.1)
        # Main velocity plot
        y = data['filtered_velocity']
        time = data['elapsed_time']
        mean = y.mean()
        std = y.std()
        ucl = mean + 3 * std
        lcl = mean - 3 * std
        out_of_control = (y > ucl) | (y < lcl)
        fig.add_trace(
            go.Scatter(x=time, y=y, name='Actual Velocity', line=dict(color='green')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=time, y=[mean]*len(y), name='Mean', line=dict(color='blue', dash='dash')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=time, y=[ucl]*len(y), name='+3σ', line=dict(color='red', dash='dot')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=time, y=[lcl]*len(y), name='-3σ', line=dict(color='red', dash='dot')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=time[out_of_control], y=y[out_of_control], mode='markers', name='Out of Control', marker=dict(color='orange', size=8, symbol='x')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=time, y=[velocity_setpoint] * len(data),
                      name='Velocity Setpoint', line=dict(color='red', dash='dash')),
            row=1, col=1
        )
        # Velocity error analysis
        velocity_error = data['filtered_velocity'] - velocity_setpoint
        fig.add_trace(
            go.Scatter(x=time, y=velocity_error,
                      name='Velocity Error', line=dict(color='orange')),
            row=2, col=1
        )
        
        # Jerk analysis (rate of change of acceleration)
        dt = data['elapsed_time'].diff().fillna(0)
        acceleration = data['filtered_velocity'].diff().fillna(0) / dt
        jerk = acceleration.diff().fillna(0) / dt
        fig.add_trace(
            go.Scatter(x=time, y=jerk,
                      name='Jerk', line=dict(color='purple')),
            row=3, col=1
        )
        
        fig.update_layout(
            title='Velocity Control Analysis',
            height=1000,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
        
    def _create_torque_plot(self, test_data: Dict) -> go.Figure:
        """Create a plot showing torque data."""
        data = test_data['data']
        metadata = test_data['metadata']
        motor_params = metadata['motor_params']
        torque_setpoint = metadata.get('torque_setpoint', 0)
        fig = make_subplots(rows=3, cols=1,
                           subplot_titles=('Torque vs Time', 'Joint Stiffness Analysis', 'Damping Analysis'),
                           vertical_spacing=0.1)
        # Main torque plot
        y = data['torque']
        time = data['elapsed_time']
        mean = y.mean()
        std = y.std()
        ucl = mean + 3 * std
        lcl = mean - 3 * std
        out_of_control = (y > ucl) | (y < lcl)
        fig.add_trace(
            go.Scatter(x=time, y=y, name='Actual Torque', line=dict(color='red')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=time, y=[mean]*len(y), name='Mean', line=dict(color='blue', dash='dash')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=time, y=[ucl]*len(y), name='+3σ', line=dict(color='green', dash='dot')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=time, y=[lcl]*len(y), name='-3σ', line=dict(color='green', dash='dot')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=time[out_of_control], y=y[out_of_control], mode='markers', name='Out of Control', marker=dict(color='orange', size=8, symbol='x')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=time, y=[torque_setpoint] * len(data),
                      name='Torque Setpoint', line=dict(color='blue', dash='dash')),
            row=1, col=1
        )
        # Joint stiffness analysis (torque vs position)
        # K = dτ/dθ (N⋅m/rad)
        # Use a moving window to calculate stiffness
        window_size = 50  # Adjust window size based on your sampling rate
        position = data['filtered_position'].values
        torque = data['torque'].values
        time = data['elapsed_time'].values
        
        # Calculate stiffness using linear regression over moving windows
        stiffness = np.zeros_like(position)
        for i in range(len(position)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(position), i + window_size // 2)
            
            if end_idx - start_idx > 1:  # Need at least 2 points for regression
                pos_window = position[start_idx:end_idx]
                torque_window = torque[start_idx:end_idx]
                
                # Only calculate stiffness if there's significant position change
                if np.std(pos_window) > 1e-3:  # Threshold for meaningful position change
                    # Linear regression
                    slope, _ = np.polyfit(pos_window, torque_window, 1)
                    stiffness[i] = slope
                else:
                    stiffness[i] = np.nan
        
        # Filter out extreme values
        stiffness = np.clip(stiffness, -1000, 1000)  # Adjust limits based on your system
        
        fig.add_trace(
            go.Scatter(x=time, y=stiffness,
                      name='Joint Stiffness', line=dict(color='green')),
            row=2, col=1
        )
        
        # Add nominal stiffness line if available
        if 'nominal_stiffness' in motor_params:
            fig.add_trace(
                go.Scatter(x=time, 
                          y=[motor_params['nominal_stiffness']] * len(time),
                          name='Nominal Stiffness', 
                          line=dict(color='gray', dash='dash')),
                row=2, col=1
            )
        
        # Damping analysis (torque vs velocity)
        # B = dτ/dω (N⋅m⋅s/rad)
        velocity = data['filtered_velocity'].values
        
        # Calculate damping using the same window approach
        damping = np.zeros_like(velocity)
        for i in range(len(velocity)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(velocity), i + window_size // 2)
            
            if end_idx - start_idx > 1:
                vel_window = velocity[start_idx:end_idx]
                torque_window = torque[start_idx:end_idx]
                
                # Only calculate damping if there's significant velocity change
                if np.std(vel_window) > 1e-3:  # Threshold for meaningful velocity change
                    # Linear regression
                    slope, _ = np.polyfit(vel_window, torque_window, 1)
                    damping[i] = slope
                else:
                    damping[i] = np.nan
        
        # Filter out extreme values
        damping = np.clip(damping, -100, 100)  # Adjust limits based on your system
        
        fig.add_trace(
            go.Scatter(x=time, y=damping,
                      name='Damping Coefficient', line=dict(color='orange')),
            row=3, col=1
        )
        
        # Add nominal damping line if available
        if 'nominal_damping' in motor_params:
            fig.add_trace(
                go.Scatter(x=time, 
                          y=[motor_params['nominal_damping']] * len(time),
                          name='Nominal Damping', 
                          line=dict(color='gray', dash='dash')),
                row=3, col=1
            )
        
        # Update y-axis labels with units
        fig.update_yaxes(title_text="Torque (N⋅m)", row=1, col=1)
        fig.update_yaxes(title_text="Stiffness (N⋅m/rad)", row=2, col=1)
        fig.update_yaxes(title_text="Damping (N⋅m⋅s/rad)", row=3, col=1)
        
        # Update x-axis labels
        fig.update_xaxes(title_text="Time (s)", row=1, col=1)
        fig.update_xaxes(title_text="Time (s)", row=2, col=1)
        fig.update_xaxes(title_text="Time (s)", row=3, col=1)
        
        fig.update_layout(
            title='Torque Control Analysis',
            height=1000,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
        
    def _create_current_plot(self, test_data: Dict) -> go.Figure:
        """Create a plot showing current data."""
        data = test_data['data']
        metadata = test_data['metadata']
        motor_params = metadata['motor_params']
        current_setpoint = metadata.get('current_setpoint', 0)
        fig = make_subplots(rows=3, cols=1,
                           subplot_titles=('Current vs Time', 'Power Analysis', 'Efficiency Analysis'),
                           vertical_spacing=0.1)
        # Main current plot
        y = data['filtered_current']
        time = data['elapsed_time']
        mean = y.mean()
        std = y.std()
        ucl = mean + 3 * std
        lcl = mean - 3 * std
        out_of_control = (y > ucl) | (y < lcl)
        fig.add_trace(
            go.Scatter(x=time, y=y, name='Actual Current', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=time, y=[mean]*len(y), name='Mean', line=dict(color='green', dash='dash')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=time, y=[ucl]*len(y), name='+3σ', line=dict(color='red', dash='dot')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=time, y=[lcl]*len(y), name='-3σ', line=dict(color='red', dash='dot')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=time[out_of_control], y=y[out_of_control], mode='markers', name='Out of Control', marker=dict(color='orange', size=8, symbol='x')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=time, y=[current_setpoint] * len(data),
                      name='Current Setpoint', line=dict(color='red', dash='dash')),
            row=1, col=1
        )
        # Power analysis
        electrical_power = data['filtered_current']**2 * motor_params['resistance']
        mechanical_power = data['torque'] * data['filtered_velocity']
        fig.add_trace(
            go.Scatter(x=time, y=electrical_power,
                      name='Electrical Power', line=dict(color='green')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=time, y=mechanical_power,
                      name='Mechanical Power', line=dict(color='orange')),
            row=2, col=1
        )
        
        # Efficiency analysis
        efficiency = mechanical_power / (electrical_power + 1e-6) * 100  # Avoid division by zero
        fig.add_trace(
            go.Scatter(x=time, y=efficiency,
                      name='Efficiency (%)', line=dict(color='purple')),
            row=3, col=1
        )
        
        fig.update_layout(
            title='Current Control Analysis',
            height=1000,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
        
    def _create_motor_params_table(self, metadata: Dict) -> html.Div:
        """Create motor parameters table."""
        if 'motor_params' not in metadata:
            return html.Div("No motor parameters available")
            
        params = metadata['motor_params']
        return html.Div([
            html.Table([
                html.Tr([
                    html.Td("Inertia:"),
                    html.Td(f"{params['inertia']:.3f} kg⋅m²")
                ]),
                html.Tr([
                    html.Td("Damping:"),
                    html.Td(f"{params['damping']:.3f} N⋅m⋅s/rad")
                ]),
                html.Tr([
                    html.Td("Torque Constant:"),
                    html.Td(f"{params['torque_constant']:.3f} N⋅m/A")
                ]),
                html.Tr([
                    html.Td("Max Torque:"),
                    html.Td(f"{params['max_torque']:.1f} N⋅m")
                ]),
                html.Tr([
                    html.Td("Max Speed:"),
                    html.Td(f"{params['max_speed']:.1f} rad/s")
                ]),
                html.Tr([
                    html.Td("Resistance:"),
                    html.Td(f"{params['resistance']:.3f} Ω")
                ]),
                html.Tr([
                    html.Td("Inductance:"),
                    html.Td(f"{params['inductance']:.3f} H")
                ]),
                html.Tr([
                    html.Td("Gear Ratio:"),
                    html.Td(f"{params['gear_ratio']:.2f}")
                ]),
                html.Tr([
                    html.Td("Gear Efficiency:"),
                    html.Td(f"{params['gear_efficiency']:.1%}")
                ])
            ], style={'margin': '20px auto'})
        ])
        
    def _create_pid_gains_table(self, metadata: Dict) -> html.Div:
        """Create PID gains table."""
        if 'pid_gains' not in metadata:
            return html.Div("No PID gains available")
            
        gains = metadata['pid_gains']
        return html.Div([
            html.Table([
                html.Tr([
                    html.Td("Proportional Gain (Kp):"),
                    html.Td(f"{gains['kp']:.3f}")
                ]),
                html.Tr([
                    html.Td("Integral Gain (Ki):"),
                    html.Td(f"{gains['ki']:.3f}")
                ]),
                html.Tr([
                    html.Td("Derivative Gain (Kd):"),
                    html.Td(f"{gains['kd']:.3f}")
                ])
            ], style={'margin': '20px auto'})
        ])
        
    def _create_sensor_config_table(self, metadata: Dict) -> html.Div:
        """Create sensor configuration table."""
        if 'sensor_params' not in metadata or metadata['sensor_params'] is None:
            return html.Div("No sensor configuration available")
            
        params = metadata['sensor_params']
        return html.Div([
            html.Table([
                html.Tr([
                    html.Td("Position Noise:"),
                    html.Td(f"{params['position_noise']:.3f} rad")
                ]),
                html.Tr([
                    html.Td("Velocity Noise:"),
                    html.Td(f"{params['velocity_noise']:.3f} rad/s")
                ]),
                html.Tr([
                    html.Td("Current Noise:"),
                    html.Td(f"{params['current_noise']:.3f} A")
                ])
            ], style={'margin': '20px auto'})
        ])
        
    def _create_motor_specs_table(self, metadata: Dict) -> html.Div:
        """Create a table showing motor specifications."""
        print("\n=== Motor Specifications ===")
        print(f"Metadata keys: {metadata.keys()}")
        
        # Try different possible locations for motor specs
        motor_specs = metadata.get('motor_specs', {})
        if not motor_specs:
            motor_specs = metadata.get('motor', {})
        if not motor_specs:
            motor_specs = metadata.get('motor_parameters', {})
            
        print(f"Motor specs found: {motor_specs}")
        
        if not motor_specs:
            return html.Div("No motor specifications available")
            
        # Extract motor type from various possible locations
        motor_type = (
            motor_specs.get('type') or 
            motor_specs.get('motor_type') or 
            motor_specs.get('model') or 
            'N/A'
        )
        
        table = html.Table([
            html.Thead(
                html.Tr([
                    html.Th("Parameter"),
                    html.Th("Value"),
                    html.Th("Unit")
                ])
            ),
            html.Tbody([
                html.Tr([
                    html.Td("Motor Type"),
                    html.Td(motor_type),
                    html.Td("")
                ]),
                html.Tr([
                    html.Td("Rated Power"),
                    html.Td(f"{motor_specs.get('rated_power', motor_specs.get('power', 0)):.1f}"),
                    html.Td("W")
                ]),
                html.Tr([
                    html.Td("Rated Torque"),
                    html.Td(f"{motor_specs.get('rated_torque', motor_specs.get('torque', 0)):.2f}"),
                    html.Td("N⋅m")
                ]),
                html.Tr([
                    html.Td("Rated Speed"),
                    html.Td(f"{motor_specs.get('rated_speed', motor_specs.get('speed', 0)):.1f}"),
                    html.Td("RPM")
                ]),
                html.Tr([
                    html.Td("Rated Current"),
                    html.Td(f"{motor_specs.get('rated_current', motor_specs.get('current', 0)):.1f}"),
                    html.Td("A")
                ]),
                html.Tr([
                    html.Td("Rated Voltage"),
                    html.Td(f"{motor_specs.get('rated_voltage', motor_specs.get('voltage', 0)):.1f}"),
                    html.Td("V")
                ]),
                html.Tr([
                    html.Td("Encoder Resolution"),
                    html.Td(f"{motor_specs.get('encoder_resolution', motor_specs.get('resolution', 0))}"),
                    html.Td("counts/rev")
                ]),
                html.Tr([
                    html.Td("Gear Ratio"),
                    html.Td(f"{motor_specs.get('gear_ratio', motor_specs.get('ratio', 1)):.1f}"),
                    html.Td("")
                ])
            ])
        ], style={'width': '100%', 'border': '1px solid black'})
        
        return table
        
    def _create_sensor_specs_table(self, metadata: Dict) -> html.Div:
        """Create a table showing sensor specifications."""
        print("\n=== Sensor Specifications ===")
        print(f"Metadata keys: {metadata.keys()}")
        
        # Try different possible locations for sensor specs
        sensor_specs = metadata.get('sensor_specs', {})
        if not sensor_specs:
            sensor_specs = metadata.get('sensors', {})
        if not sensor_specs:
            sensor_specs = metadata.get('sensor_parameters', {})
            
        print(f"Sensor specs found: {sensor_specs}")
        
        if not sensor_specs:
            return html.Div("No sensor specifications available")
            
        # Helper function to get sensor type
        def get_sensor_type(sensor_data):
            return (
                sensor_data.get('type') or
                sensor_data.get('sensor_type') or
                sensor_data.get('model') or
                'N/A'
            )
            
        # Helper function to get sensor specs
        def get_sensor_specs(sensor_name):
            sensor_data = sensor_specs.get(sensor_name, {})
            if not sensor_data:
                # Try alternative naming
                alt_name = {
                    'position': ['pos', 'encoder', 'angle'],
                    'velocity': ['vel', 'speed', 'tachometer'],
                    'current': ['curr', 'amp', 'current_sensor'],
                    'torque': ['torque', 'force', 'load_cell']
                }.get(sensor_name, [])
                
                for name in alt_name:
                    if name in sensor_specs:
                        sensor_data = sensor_specs[name]
                        break
                        
            return sensor_data
            
        table = html.Table([
            html.Thead(
                html.Tr([
                    html.Th("Sensor Type"),
                    html.Th("Parameter"),
                    html.Th("Value"),
                    html.Th("Unit")
                ])
            ),
            html.Tbody([
                # Position Sensor
                html.Tr([
                    html.Td("Position Sensor"),
                    html.Td("Type"),
                    html.Td(get_sensor_type(get_sensor_specs('position'))),
                    html.Td("")
                ]),
                html.Tr([
                    html.Td(""),
                    html.Td("Resolution"),
                    html.Td(f"{get_sensor_specs('position').get('resolution', 0):.3f}"),
                    html.Td("rad")
                ]),
                html.Tr([
                    html.Td(""),
                    html.Td("Range"),
                    html.Td(f"±{get_sensor_specs('position').get('range', 0):.1f}"),
                    html.Td("rad")
                ]),
                
                # Velocity Sensor
                html.Tr([
                    html.Td("Velocity Sensor"),
                    html.Td("Type"),
                    html.Td(get_sensor_type(get_sensor_specs('velocity'))),
                    html.Td("")
                ]),
                html.Tr([
                    html.Td(""),
                    html.Td("Resolution"),
                    html.Td(f"{get_sensor_specs('velocity').get('resolution', 0):.3f}"),
                    html.Td("rad/s")
                ]),
                html.Tr([
                    html.Td(""),
                    html.Td("Range"),
                    html.Td(f"±{get_sensor_specs('velocity').get('range', 0):.1f}"),
                    html.Td("rad/s")
                ]),
                
                # Current Sensor
                html.Tr([
                    html.Td("Current Sensor"),
                    html.Td("Type"),
                    html.Td(get_sensor_type(get_sensor_specs('current'))),
                    html.Td("")
                ]),
                html.Tr([
                    html.Td(""),
                    html.Td("Resolution"),
                    html.Td(f"{get_sensor_specs('current').get('resolution', 0):.3f}"),
                    html.Td("A")
                ]),
                html.Tr([
                    html.Td(""),
                    html.Td("Range"),
                    html.Td(f"±{get_sensor_specs('current').get('range', 0):.1f}"),
                    html.Td("A")
                ]),
                
                # Torque Sensor
                html.Tr([
                    html.Td("Torque Sensor"),
                    html.Td("Type"),
                    html.Td(get_sensor_type(get_sensor_specs('torque'))),
                    html.Td("")
                ]),
                html.Tr([
                    html.Td(""),
                    html.Td("Resolution"),
                    html.Td(f"{get_sensor_specs('torque').get('resolution', 0):.3f}"),
                    html.Td("N⋅m")
                ]),
                html.Tr([
                    html.Td(""),
                    html.Td("Range"),
                    html.Td(f"±{get_sensor_specs('torque').get('range', 0):.1f}"),
                    html.Td("N⋅m")
                ])
            ])
        ], style={'width': '100%', 'border': '1px solid black'})
        
        return table
        
    def _create_config_comparison_table(self, test_data_list: List[Tuple[str, Dict]]) -> html.Div:
        """Create a table comparing configurations across tests."""
        rows = []
        
        # Add header row
        header_cells = [html.Th("Parameter")] + [html.Th(test_name) for test_name, _ in test_data_list]
        rows.append(html.Tr(header_cells))
        
        # Compare PID gains
        for gain in ['kp', 'ki', 'kd']:
            cells = [html.Td(f"PID {gain.upper()}")]
            for _, test_data in test_data_list:
                value = test_data['metadata']['pid_gains'][gain]
                cells.append(html.Td(f"{value:.3f}"))
            rows.append(html.Tr(cells))
            
        # Compare motor parameters
        for param in ['inertia', 'damping', 'torque_constant']:
            cells = [html.Td(param.replace('_', ' ').title())]
            for _, test_data in test_data_list:
                value = test_data['metadata']['motor_params'][param]
                cells.append(html.Td(f"{value:.3f}"))
            rows.append(html.Tr(cells))
            
        return html.Table(rows, style={'margin': '20px auto', 'border': '1px solid black'})
        
    def _create_metrics_comparison_table(self, test_data_list: List[Tuple[str, Dict]]) -> html.Div:
        """Create a table comparing performance metrics across tests."""
        rows = []
        
        # Add header row
        header_cells = [html.Th("Metric")] + [html.Th(test_name) for test_name, _ in test_data_list]
        rows.append(html.Tr(header_cells))
        
        # Compare metrics
        metrics = [
            ('Rise Time', 'rise_time', 's'),
            ('Settling Time', 'settling_time', 's'),
            ('Overshoot', 'overshoot', '%'),
            ('Steady State Error', 'steady_state_error', 'rad'),
            ('RMS Error', 'rms_error', 'rad'),
            ('Peak Torque', 'peak_torque', 'N⋅m'),
            ('Peak Velocity', 'peak_velocity', 'rad/s')
        ]
        
        for metric_name, metric_key, unit in metrics:
            cells = [html.Td(metric_name)]
            for _, test_data in test_data_list:
                analyzer = Analyzer(test_data['data'], test_data['metadata']['setpoint'])
                metrics = analyzer.compute_metrics()
                value = getattr(metrics, metric_key)
                cells.append(html.Td(f"{value:.3f} {unit}"))
            rows.append(html.Tr(cells))
            
        return html.Table(rows, style={'margin': '20px auto', 'border': '1px solid black'})
        
    def _create_ab_testing_plot(self, test_data_list: List[Tuple[str, Dict]]) -> go.Figure:
        """Create an A/B testing visualization plot."""
        fig = go.Figure()
        
        # Plot position responses for all tests
        for test_name, test_data in test_data_list:
            data = test_data['data']
            setpoint = test_data['metadata']['setpoint']
            
            # Add position trace
            fig.add_trace(go.Scatter(
                x=data['elapsed_time'],
                y=data['filtered_position'],
                name=f"{test_name} - Position",
                line=dict(width=2)
            ))
            
            # Add setpoint line
            fig.add_trace(go.Scatter(
                x=data['elapsed_time'],
                y=[setpoint] * len(data),
                name=f"{test_name} - Setpoint",
                line=dict(dash='dash', width=1)
            ))
            
        # Update layout
        fig.update_layout(
            title='Position Response Comparison',
            xaxis_title='Time (s)',
            yaxis_title='Position (rad)',
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
        
    def _create_sensor_comparison_plot(self, test_data: Dict) -> go.Figure:
        """Create a plot comparing raw and filtered sensor data."""
        data = test_data['data']
        
        fig = make_subplots(rows=4, cols=1,
                           subplot_titles=('Position', 'Velocity', 'Current', 'Torque'),
                           vertical_spacing=0.1)
        
        # Position comparison
        fig.add_trace(
            go.Scatter(x=data['elapsed_time'], y=data['raw_position'],
                      name='Raw Position', line=dict(color='blue', dash='dot')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=data['elapsed_time'], y=data['filtered_position'],
                      name='Filtered Position', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Velocity comparison
        fig.add_trace(
            go.Scatter(x=data['elapsed_time'], y=data['raw_velocity'],
                      name='Raw Velocity', line=dict(color='green', dash='dot')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=data['elapsed_time'], y=data['filtered_velocity'],
                      name='Filtered Velocity', line=dict(color='green')),
            row=2, col=1
        )
        
        # Current comparison
        fig.add_trace(
            go.Scatter(x=data['elapsed_time'], y=data['raw_current'],
                      name='Raw Current', line=dict(color='red', dash='dot')),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=data['elapsed_time'], y=data['filtered_current'],
                      name='Filtered Current', line=dict(color='red')),
            row=3, col=1
        )
        
        # Torque comparison
        fig.add_trace(
            go.Scatter(x=data['elapsed_time'], y=data['torque'],
                      name='Torque', line=dict(color='orange')),
            row=4, col=1
        )
        
        # Update y-axis labels with units
        fig.update_yaxes(title_text="Position (rad)", row=1, col=1)
        fig.update_yaxes(title_text="Velocity (rad/s)", row=2, col=1)
        fig.update_yaxes(title_text="Current (A)", row=3, col=1)
        fig.update_yaxes(title_text="Torque (N⋅m)", row=4, col=1)
        
        # Update x-axis labels
        fig.update_xaxes(title_text="Time (s)", row=1, col=1)
        fig.update_xaxes(title_text="Time (s)", row=2, col=1)
        fig.update_xaxes(title_text="Time (s)", row=3, col=1)
        fig.update_xaxes(title_text="Time (s)", row=4, col=1)
        
        fig.update_layout(
            title='Sensor Data Analysis',
            height=1200,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
        
    def _create_sensor_noise_plot(self, test_data: Dict) -> go.Figure:
        """Create a plot showing sensor noise analysis."""
        data = test_data['data']
        
        # Calculate noise (difference between raw and filtered)
        position_noise = data['raw_position'] - data['filtered_position']
        velocity_noise = data['raw_velocity'] - data['filtered_velocity']
        current_noise = data['raw_current'] - data['filtered_current']
        
        fig = make_subplots(rows=3, cols=1,
                           subplot_titles=('Position Noise', 'Velocity Noise', 'Current Noise'),
                           vertical_spacing=0.1)
        
        # Position noise
        fig.add_trace(
            go.Scatter(x=data['elapsed_time'], y=position_noise,
                      name='Position Noise', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Velocity noise
        fig.add_trace(
            go.Scatter(x=data['elapsed_time'], y=velocity_noise,
                      name='Velocity Noise', line=dict(color='green')),
            row=2, col=1
        )
        
        # Current noise
        fig.add_trace(
            go.Scatter(x=data['elapsed_time'], y=current_noise,
                      name='Current Noise', line=dict(color='red')),
            row=3, col=1
        )
        
        # Update y-axis labels with units
        fig.update_yaxes(title_text="Position Noise (rad)", row=1, col=1)
        fig.update_yaxes(title_text="Velocity Noise (rad/s)", row=2, col=1)
        fig.update_yaxes(title_text="Current Noise (A)", row=3, col=1)
        
        # Update x-axis labels
        fig.update_xaxes(title_text="Time (s)", row=1, col=1)
        fig.update_xaxes(title_text="Time (s)", row=2, col=1)
        fig.update_xaxes(title_text="Time (s)", row=3, col=1)
        
        fig.update_layout(
            title='Sensor Noise Analysis',
            height=900,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
        
    def _create_temperature_plot(self, test_data: Dict) -> go.Figure:
        """Create a plot showing temperature profile."""
        data = test_data['data']
        motor_params = test_data['metadata']['motor_params']
        
        # Calculate power dissipation
        power = data['filtered_current']**2 * motor_params['resistance']
        
        # Simple thermal model
        thermal_mass = motor_params['thermal_mass']
        thermal_resistance = motor_params['thermal_resistance']
        ambient_temp = motor_params['ambient_temp']
        
        # Calculate temperature rise
        dt = data['elapsed_time'].diff().fillna(0)
        temp_rise = np.zeros_like(data['elapsed_time'])
        temp = ambient_temp
        
        for i in range(len(data)):
            if i > 0:
                # Heat generation
                heat_gen = power[i] * dt[i]
                # Heat dissipation
                heat_diss = (temp - ambient_temp) / thermal_resistance * dt[i]
                # Temperature change
                temp += (heat_gen - heat_diss) / thermal_mass
            temp_rise[i] = temp
            
        fig = go.Figure()
        
        # Add temperature trace
        fig.add_trace(go.Scatter(
            x=data['elapsed_time'],
            y=temp_rise,
            name='Motor Temperature',
            line=dict(color='red')
        ))
        
        # Add ambient temperature line
        fig.add_trace(go.Scatter(
            x=data['elapsed_time'],
            y=[ambient_temp] * len(data),
            name='Ambient Temperature',
            line=dict(color='gray', dash='dash')
        ))
        
        # Add max temperature line
        fig.add_trace(go.Scatter(
            x=data['elapsed_time'],
            y=[motor_params['max_temp']] * len(data),
            name='Max Temperature',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title='Motor Temperature Profile',
            xaxis_title='Time (s)',
            yaxis_title='Temperature (°C)',
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
        
    def _create_thermal_model_plot(self, test_data: Dict) -> go.Figure:
        """Create a plot showing thermal model analysis."""
        import plotly.graph_objs as go
        data = test_data['data']
        motor_params = test_data['metadata']['motor_params']
        time = data['elapsed_time']
        # Calculate power dissipation
        electrical_power = data['filtered_current']**2 * motor_params['resistance']
        mechanical_power = data['torque'] * data['filtered_velocity']
        power_loss = electrical_power - mechanical_power
        # Calculate efficiency
        efficiency = mechanical_power / (electrical_power + 1e-6) * 100
        # SPC overlays for power dissipation
        mean_power = electrical_power.mean()
        std_power = electrical_power.std()
        ucl_power = mean_power + 3 * std_power
        lcl_power = mean_power - 3 * std_power
        out_of_control_power = (electrical_power > ucl_power) | (electrical_power < lcl_power)
        # SPC overlays for efficiency
        mean_eff = efficiency.mean()
        std_eff = efficiency.std()
        ucl_eff = mean_eff + 3 * std_eff
        lcl_eff = mean_eff - 3 * std_eff
        out_of_control_eff = (efficiency > ucl_eff) | (efficiency < lcl_eff)
        fig = make_subplots(rows=2, cols=1,
                           subplot_titles=('Power Dissipation', 'Motor Efficiency'),
                           vertical_spacing=0.1)
        # Power Dissipation
        fig.add_trace(go.Scatter(x=time, y=electrical_power, name='Electrical Power', line=dict(color='orange')), row=1, col=1)
        fig.add_trace(go.Scatter(x=time, y=mechanical_power, name='Mechanical Power', line=dict(color='green')), row=1, col=1)
        fig.add_trace(go.Scatter(x=time, y=power_loss, name='Power Loss', line=dict(color='red', dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=time, y=[mean_power]*len(time), name='Mean', line=dict(color='blue', dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=time, y=[ucl_power]*len(time), name='+3σ', line=dict(color='purple', dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=time[out_of_control_power], y=electrical_power[out_of_control_power], mode='markers', name='Out of Control', marker=dict(color='black', size=8, symbol='x')), row=1, col=1)
        fig.update_yaxes(title_text="Power (W)", row=1, col=1)
        fig.update_xaxes(title_text="Time (s)", row=1, col=1)
        # Efficiency
        fig.add_trace(go.Scatter(x=time, y=efficiency, name='Efficiency (%)', line=dict(color='blue')), row=2, col=1)
        fig.add_trace(go.Scatter(x=time, y=[100]*len(time), name='100% Reference', line=dict(color='gray', dash='dash')), row=2, col=1)
        fig.add_trace(go.Scatter(x=time, y=[80]*len(time), name='80% Reference', line=dict(color='green', dash='dot')), row=2, col=1)
        fig.add_trace(go.Scatter(x=time, y=[mean_eff]*len(time), name='Mean', line=dict(color='orange', dash='dash')), row=2, col=1)
        fig.add_trace(go.Scatter(x=time, y=[ucl_eff]*len(time), name='+3σ', line=dict(color='red', dash='dot')), row=2, col=1)
        fig.add_trace(go.Scatter(x=time, y=[lcl_eff]*len(time), name='-3σ', line=dict(color='red', dash='dot')), row=2, col=1)
        fig.add_trace(go.Scatter(x=time[out_of_control_eff], y=efficiency[out_of_control_eff], mode='markers', name='Out of Control', marker=dict(color='black', size=8, symbol='x')), row=2, col=1)
        fig.update_yaxes(title_text="Efficiency (%)", row=2, col=1)
        fig.update_xaxes(title_text="Time (s)", row=2, col=1)
        fig.update_layout(
            title='Thermal Model Analysis',
            height=700,
            showlegend=True,
            hovermode='x unified'
        )
        return fig
        
    def run_server(self, debug: bool = True, port: int = 8050):
        """Run the dashboard server."""
        print(f"\nStarting dashboard server on port {port}")
        self.app.run(debug=debug, port=port) 