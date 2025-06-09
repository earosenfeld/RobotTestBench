"""
Simple test dashboard for RobotTestBench with sensor and dyno data.
"""

import dash
from dash import html, dcc
import plotly.graph_objs as go
import numpy as np
from simulation.sensors import SensorSimulator, SensorParameters
from simulation.dyno import DynoSimulator, DynoParameters
from simulation.motor import MotorParameters
from robot_testbench.sensors import (
    QuadratureEncoder, QuadratureEncoderConfig,
    ForceTorqueSensor, ForceTorqueSensorConfig,
    JointAngleSensor, JointAngleSensorConfig
)

# Initialize the Dash app
app = dash.Dash(__name__)

# Create sensor simulator
sensor_params = SensorParameters(
    position_noise_std=0.001,  # rad
    velocity_noise_std=0.01,   # rad/s
    current_noise_std=0.1,     # A
    sampling_rate=1000.0,      # Hz
    filter_type='lowpass',     # Filter type
    filter_cutoff=50.0         # Hz
)
sensor = SensorSimulator(sensor_params)

# Create motor parameters
motor_params = MotorParameters(
    inertia=0.05,  # kg⋅m²
    damping=1.0,   # N⋅m⋅s/rad
    torque_constant=0.1,  # N⋅m/A
    max_torque=4.0,  # N⋅m
    max_speed=20.0,  # rad/s
    resistance=1.0,  # Ω
    inductance=0.1   # H
)

# Create dyno simulator
dyno_params = DynoParameters(
    drive_motor=motor_params,
    load_motor=motor_params,
    coupling_stiffness=1000.0,  # N⋅m/rad
    coupling_damping=0.1,       # N⋅m⋅s/rad
    max_torque_transfer=10.0    # N⋅m
)
dyno = DynoSimulator(dyno_params)

# Generate test data
t = np.linspace(0, 10, 1000)  # 10 seconds at 100 Hz
dt = t[1] - t[0]  # s

# Generate motor setpoints
position_setpoint = np.sin(t)  # rad
velocity_setpoint = np.cos(t)  # rad/s
torque_setpoint = 0.5 * np.sin(t) * np.cos(t)  # N⋅m

# Simulate sensor and dyno data
sensor_positions = []  # rad
sensor_velocities = []  # rad/s
sensor_currents = []  # A
dyno_positions = []  # rad
dyno_velocities = []  # rad/s
dyno_torques = []  # N⋅m

# --- New sensor model instantiations ---
encoder_config = QuadratureEncoderConfig(
    resolution=1024,  # 1024 counts/rev
    noise_std=1.0,
    edge_trigger_noise=0.0001,
    max_frequency=1000.0
)
encoder = QuadratureEncoder(encoder_config)

ft_sensor_config = ForceTorqueSensorConfig(
    sensitivity=1.0,  # 1 N⋅m/V
    noise_std=0.05,   # 50 mV noise
    drift_rate=0.005, # 5 mV/s drift
    hysteresis=0.05,  # 50 mN⋅m
    calibration_offset=0.01,  # 10 mV offset
    temperature_coefficient=0.0005
)
ft_sensor = ForceTorqueSensor(ft_sensor_config)

ja_sensor_config = JointAngleSensorConfig(
    resolution=0.001,  # 1 mrad
    noise_std=0.0005,  # 0.5 mrad
    backlash=0.01,     # 10 mrad
    limit_stops=(-np.pi, np.pi),
    calibration_offset=0.002,  # 2 mrad offset
    temperature_coefficient=0.0001
)
ja_sensor = JointAngleSensor(ja_sensor_config)

# --- Data arrays for new sensors ---
encoder_a = []
encoder_b = []
ft_voltages = []
ja_angles = []

for i in range(len(t)):
    # Get sensor data
    (raw_pos, raw_vel, raw_curr), (filt_pos, filt_vel, filt_curr) = sensor.process_signals(
        position_setpoint[i], velocity_setpoint[i], torque_setpoint[i]
    )
    sensor_positions.append(filt_pos)  # rad
    sensor_velocities.append(filt_vel)  # rad/s
    sensor_currents.append(filt_curr)  # A
    
    # Get dyno data
    drive_state, load_state = dyno.step(torque_setpoint[i], 0.0, dt)  # No voltage to load motor
    dyno_positions.append(drive_state[0])  # rad
    dyno_velocities.append(drive_state[1])  # rad/s
    dyno_torques.append(drive_state[2])  # N⋅m

    # New sensor models
    a, b = encoder.update(position_setpoint[i], velocity_setpoint[i], dt)
    encoder_a.append(a)
    encoder_b.append(b)
    ft_voltages.append(ft_sensor.update(torque_setpoint[i], dt))
    ja_angles.append(ja_sensor.update(position_setpoint[i], velocity_setpoint[i]))

# Convert to numpy arrays
sensor_positions = np.array(sensor_positions)  # rad
sensor_velocities = np.array(sensor_velocities)  # rad/s
sensor_currents = np.array(sensor_currents)  # A
dyno_positions = np.array(dyno_positions)  # rad
dyno_velocities = np.array(dyno_velocities)  # rad/s
dyno_torques = np.array(dyno_torques)  # N⋅m
encoder_a = np.array(encoder_a)
encoder_b = np.array(encoder_b) - 0.2  # Offset B channel to 0.8/ -0.2 for visual separation
ft_voltages = np.array(ft_voltages)
ja_angles = np.array(ja_angles)

# Common layout settings
common_layout = {
    'paper_bgcolor': 'rgba(0,0,0,0)',
    'plot_bgcolor': 'rgba(0,0,0,0)',
    'font': {'family': 'Arial, sans-serif'},
    'margin': {'l': 50, 'r': 20, 't': 50, 'b': 50},
    'showlegend': True,
    'legend': {
        'orientation': 'h',
        'yanchor': 'bottom',
        'y': 1.02,
        'xanchor': 'right',
        'x': 1
    }
}

# X-axis settings
xaxis_settings = {
    'title': {
        'text': 'Time [s] (10s total, 100 Hz sampling)',
        'font': {'size': 12}
    },
    'gridcolor': '#e0e0e0',
    'showgrid': True,
    'zeroline': True,
    'zerolinecolor': '#969696',
    'zerolinewidth': 1,
    'tickformat': '.1f',  # Show one decimal place
    'dtick': 1.0,  # Show ticks every second
    'range': [0, 10]  # Fixed range from 0 to 10 seconds
}

# Common y-axis settings
yaxis_common = {
    'gridcolor': '#e0e0e0',
    'showgrid': True,
    'zeroline': True,
    'zerolinecolor': '#969696',
    'zerolinewidth': 1,
    'title': {
        'font': {'size': 14},
        'standoff': 10
    }
}

# Create the layout
app.layout = html.Div([
    html.H1("RobotTestBench Test Dashboard", 
            style={'textAlign': 'center', 'color': '#2c3e50', 'margin': '20px'}),
    
    # Position plot
    dcc.Graph(
        id='position-plot',
        figure={
            'data': [
                {
                    'x': t,
                    'y': position_setpoint,
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': 'Position Setpoint',
                    'line': {'width': 2, 'color': '#1f77b4'}
                },
                {
                    'x': t,
                    'y': sensor_positions,
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': 'Sensor Position',
                    'line': {'width': 2, 'color': '#ff7f0e'}
                },
                {
                    'x': t,
                    'y': dyno_positions,
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': 'Dyno Position',
                    'line': {'width': 2, 'color': '#2ca02c'}
                }
            ],
            'layout': {
                **common_layout,
                'title': {
                    'text': 'Motor Position vs Time',
                    'font': {'size': 24, 'color': '#2c3e50'},
                    'y': 0.95
                },
                'xaxis': xaxis_settings,
                'yaxis': {
                    **yaxis_common,
                    'title': {
                        'text': 'Position [rad]',
                        'font': {'size': 14},
                        'standoff': 10
                    }
                }
            }
        },
        style={'height': '400px', 'margin': '20px'}
    ),
    
    # Velocity plot
    dcc.Graph(
        id='velocity-plot',
        figure={
            'data': [
                {
                    'x': t,
                    'y': velocity_setpoint,
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': 'Velocity Setpoint',
                    'line': {'width': 2, 'color': '#1f77b4'}
                },
                {
                    'x': t,
                    'y': sensor_velocities,
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': 'Sensor Velocity',
                    'line': {'width': 2, 'color': '#ff7f0e'}
                },
                {
                    'x': t,
                    'y': dyno_velocities,
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': 'Dyno Velocity',
                    'line': {'width': 2, 'color': '#2ca02c'}
                }
            ],
            'layout': {
                **common_layout,
                'title': {
                    'text': 'Motor Velocity vs Time',
                    'font': {'size': 24, 'color': '#2c3e50'},
                    'y': 0.95
                },
                'xaxis': xaxis_settings,
                'yaxis': {
                    **yaxis_common,
                    'title': {
                        'text': 'Velocity [rad/s]',
                        'font': {'size': 14},
                        'standoff': 10
                    }
                }
            }
        },
        style={'height': '400px', 'margin': '20px'}
    ),
    
    # Torque/Current plot
    dcc.Graph(
        id='torque-plot',
        figure={
            'data': [
                {
                    'x': t,
                    'y': torque_setpoint,
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': 'Torque Setpoint',
                    'line': {'width': 2, 'color': '#1f77b4'}
                },
                {
                    'x': t,
                    'y': sensor_currents,
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': 'Sensor Current',
                    'line': {'width': 2, 'color': '#ff7f0e'}
                },
                {
                    'x': t,
                    'y': dyno_torques,
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': 'Dyno Torque',
                    'line': {'width': 2, 'color': '#2ca02c'}
                }
            ],
            'layout': {
                **common_layout,
                'title': {
                    'text': 'Motor Torque and Current vs Time',
                    'font': {'size': 24, 'color': '#2c3e50'},
                    'y': 0.95
                },
                'xaxis': xaxis_settings,
                'yaxis': {
                    **yaxis_common,
                    'title': {
                        'text': 'Torque [N⋅m] / Current [A]',
                        'font': {'size': 14},
                        'standoff': 10
                    }
                }
            }
        },
        style={'height': '400px', 'margin': '20px'}
    ),
    # Quadrature Encoder Channels
    dcc.Graph(
        id='encoder-plot',
        figure={
            'data': [
                {'x': t, 'y': encoder_a, 'type': 'scatter', 'mode': 'lines+markers', 'name': 'Encoder A (Logic High/Low)', 'line': {'width': 2, 'color': '#d62728', 'shape': 'hv'}},
                {'x': t, 'y': encoder_b, 'type': 'scatter', 'mode': 'lines+markers', 'name': 'Encoder B (Logic High/Low, offset)', 'line': {'width': 2, 'color': '#9467bd', 'shape': 'hv'}},
            ],
            'layout': {
                **common_layout,
                'title': {'text': 'Quadrature Encoder Channels vs Time', 'font': {'size': 22, 'color': '#2c3e50'}, 'y': 0.95},
                'xaxis': xaxis_settings,
                'yaxis': {
                    **yaxis_common,
                    'title': {
                        'text': 'Logic Level (0 = Low, 1 = High)\nB channel offset for clarity',
                        'font': {'size': 14},
                        'standoff': 10
                    },
                    'dtick': 0.2,
                    'range': [-0.3, 1.2]
                }
            }
        },
        style={'height': '250px', 'margin': '20px'}
    ),
    # Force/Torque Sensor Output
    dcc.Graph(
        id='ft-sensor-plot',
        figure={
            'data': [
                {'x': t, 'y': ft_voltages, 'type': 'scatter', 'mode': 'lines', 'name': 'Force/Torque Sensor Voltage', 'line': {'width': 2, 'color': '#8c564b'}},
            ],
            'layout': {
                **common_layout,
                'title': {'text': 'Force/Torque Sensor Output vs Time', 'font': {'size': 22, 'color': '#2c3e50'}, 'y': 0.95},
                'xaxis': xaxis_settings,
                'yaxis': {
                    **yaxis_common,
                    'title': {
                        'text': 'Sensor Output [V]\n(Includes noise, drift, hysteresis)',
                        'font': {'size': 14},
                        'standoff': 10
                    }
                }
            }
        },
        style={'height': '250px', 'margin': '20px'}
    ),
    # Joint Angle Sensor Output
    dcc.Graph(
        id='ja-sensor-plot',
        figure={
            'data': [
                {'x': t, 'y': ja_angles, 'type': 'scatter', 'mode': 'lines', 'name': 'Joint Angle Sensor', 'line': {'width': 2, 'color': '#e377c2'}},
            ],
            'layout': {
                **common_layout,
                'title': {'text': 'Joint Angle Sensor Output vs Time', 'font': {'size': 22, 'color': '#2c3e50'}, 'y': 0.95},
                'xaxis': xaxis_settings,
                'yaxis': {
                    **yaxis_common,
                    'title': {
                        'text': 'Angle [rad]\n(Quantized, noisy, with backlash)',
                        'font': {'size': 14},
                        'standoff': 10
                    }
                }
            }
        },
        style={'height': '250px', 'margin': '20px'}
    ),
], style={'backgroundColor': '#f8f9fa', 'padding': '20px'})

if __name__ == '__main__':
    print("Starting dashboard...")
    print("Open your browser and go to http://127.0.0.1:8050")
    app.run(debug=True, port=8050) 