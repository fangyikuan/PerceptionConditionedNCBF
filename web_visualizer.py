"""Web visualizer for DrivingContinuousRandom environment.

This script creates a web server that allows users to visualize the DrivingContinuousRandom
environment with different obstacle configurations.
"""

import posggym
import numpy as np
import time
import base64
import io
from PIL import Image
import threading
import argparse
from flask import Flask, render_template_string, request, jsonify

# Global variables
env = None
running = False
thread = None
frame_buffer = []
step_count = 0
episode_count = 0

# Create Flask app
app = Flask(__name__)

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>DrivingContinuousRandom Visualizer</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .container {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        .controls {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            background-color: #fff;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .control-group {
            flex: 1;
            min-width: 200px;
        }
        .visualization {
            background-color: #fff;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            text-align: center;
        }
        .info {
            background-color: #fff;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 15px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            margin-top: 10px;
        }
        button:hover {
            background-color: #45a049;
        }
        button:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
        }
        input, select {
            width: 100%;
            padding: 8px;
            margin: 5px 0;
            display: inline-block;
            border: 1px solid #ccc;
            border-radius: 4px;
            box-sizing: border-box;
        }
        label {
            font-weight: bold;
        }
        #status {
            font-weight: bold;
            color: #4CAF50;
        }
        #visualization-container {
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 400px;
        }
        #env-image {
            max-width: 100%;
            max-height: 400px;
            border: 1px solid #ddd;
        }
        .info-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }
        .info-item {
            padding: 10px;
            background-color: #f9f9f9;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <h1>DrivingContinuousRandom Environment Visualizer</h1>
    
    <div class="container">
        <div class="controls">
            <div class="control-group">
                <h3>Environment Configuration</h3>
                <label for="obstacle-density">Obstacle Density:</label>
                <input type="range" id="obstacle-density" name="obstacle-density" min="0" max="0.5" step="0.05" value="0.2">
                <span id="obstacle-density-value">0.2</span>
                
                <label for="min-radius">Min Obstacle Radius:</label>
                <input type="range" id="min-radius" name="min-radius" min="0.1" max="0.5" step="0.05" value="0.2">
                <span id="min-radius-value">0.2</span>
                
                <label for="max-radius">Max Obstacle Radius:</label>
                <input type="range" id="max-radius" name="max-radius" min="0.3" max="1.0" step="0.05" value="0.7">
                <span id="max-radius-value">0.7</span>
                
                <label for="num-agents">Number of Agents:</label>
                <select id="num-agents" name="num-agents">
                    <option value="2">2</option>
                    <option value="3">3</option>
                    <option value="4">4</option>
                </select>
                
                <label for="random-seed">Random Seed:</label>
                <input type="number" id="random-seed" name="random-seed" min="0" value="42">
            </div>
            
            <div class="control-group">
                <h3>Simulation Controls</h3>
                <label for="simulation-speed">Simulation Speed:</label>
                <input type="range" id="simulation-speed" name="simulation-speed" min="1" max="20" step="1" value="10">
                <span id="simulation-speed-value">10 FPS</span>
                
                <button id="start-button" onclick="startSimulation()">Start Simulation</button>
                <button id="stop-button" onclick="stopSimulation()" disabled>Stop Simulation</button>
                <button id="reset-button" onclick="resetEnvironment()">Reset Environment</button>
                
                <p id="status">Ready</p>
            </div>
        </div>
        
        <div class="visualization">
            <h3>Environment Visualization</h3>
            <div id="visualization-container">
                <img id="env-image" src="" alt="Environment visualization">
            </div>
        </div>
        
        <div class="info">
            <h3>Simulation Information</h3>
            <div class="info-grid">
                <div class="info-item">
                    <strong>Steps:</strong> <span id="step-count">0</span>
                </div>
                <div class="info-item">
                    <strong>Episodes:</strong> <span id="episode-count">0</span>
                </div>
                <div class="info-item">
                    <strong>Rewards:</strong> <span id="rewards">-</span>
                </div>
                <div class="info-item">
                    <strong>Terminations:</strong> <span id="terminations">-</span>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Update slider value displays
        document.getElementById('obstacle-density').addEventListener('input', function() {
            document.getElementById('obstacle-density-value').textContent = this.value;
        });
        
        document.getElementById('min-radius').addEventListener('input', function() {
            document.getElementById('min-radius-value').textContent = this.value;
            // Ensure max radius is always greater than min radius
            const minRadius = parseFloat(this.value);
            const maxRadiusInput = document.getElementById('max-radius');
            if (parseFloat(maxRadiusInput.value) <= minRadius) {
                maxRadiusInput.value = (minRadius + 0.1).toFixed(1);
                document.getElementById('max-radius-value').textContent = maxRadiusInput.value;
            }
        });
        
        document.getElementById('max-radius').addEventListener('input', function() {
            document.getElementById('max-radius-value').textContent = this.value;
            // Ensure min radius is always less than max radius
            const maxRadius = parseFloat(this.value);
            const minRadiusInput = document.getElementById('min-radius');
            if (parseFloat(minRadiusInput.value) >= maxRadius) {
                minRadiusInput.value = (maxRadius - 0.1).toFixed(1);
                document.getElementById('min-radius-value').textContent = minRadiusInput.value;
            }
        });
        
        document.getElementById('simulation-speed').addEventListener('input', function() {
            document.getElementById('simulation-speed-value').textContent = this.value + ' FPS';
        });
        
        // Simulation control functions
        function startSimulation() {
            const obstacleDensity = document.getElementById('obstacle-density').value;
            const minRadius = document.getElementById('min-radius').value;
            const maxRadius = document.getElementById('max-radius').value;
            const numAgents = document.getElementById('num-agents').value;
            const randomSeed = document.getElementById('random-seed').value;
            const simulationSpeed = document.getElementById('simulation-speed').value;
            
            document.getElementById('start-button').disabled = true;
            document.getElementById('stop-button').disabled = false;
            document.getElementById('status').textContent = 'Running...';
            
            // Start the simulation
            fetch('/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    obstacle_density: parseFloat(obstacleDensity),
                    min_radius: parseFloat(minRadius),
                    max_radius: parseFloat(maxRadius),
                    num_agents: parseInt(numAgents),
                    random_seed: parseInt(randomSeed),
                    simulation_speed: parseInt(simulationSpeed)
                }),
            })
            .then(response => response.json())
            .then(data => {
                if (!data.success) {
                    document.getElementById('status').textContent = 'Error: ' + data.message;
                    document.getElementById('start-button').disabled = false;
                    document.getElementById('stop-button').disabled = true;
                }
            });
            
            // Start polling for frames
            startFramePolling();
        }
        
        function stopSimulation() {
            fetch('/stop', {
                method: 'POST',
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('start-button').disabled = false;
                document.getElementById('stop-button').disabled = true;
                document.getElementById('status').textContent = 'Stopped';
            });
        }
        
        function resetEnvironment() {
            fetch('/reset', {
                method: 'POST',
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('status').textContent = 'Reset';
                document.getElementById('step-count').textContent = '0';
                document.getElementById('episode-count').textContent = '0';
                document.getElementById('rewards').textContent = '-';
                document.getElementById('terminations').textContent = '-';
            });
        }
        
        // Frame polling
        let framePollingInterval;
        
        function startFramePolling() {
            // Clear any existing interval
            if (framePollingInterval) {
                clearInterval(framePollingInterval);
            }
            
            // Set polling interval based on simulation speed
            const fps = parseInt(document.getElementById('simulation-speed').value);
            const interval = 1000 / fps;
            
            framePollingInterval = setInterval(fetchFrame, interval);
        }
        
        function fetchFrame() {
            fetch('/frame')
            .then(response => response.json())
            .then(data => {
                if (data.frame) {
                    document.getElementById('env-image').src = 'data:image/png;base64,' + data.frame;
                }
                
                if (data.step_count !== undefined) {
                    document.getElementById('step-count').textContent = data.step_count;
                }
                
                if (data.episode_count !== undefined) {
                    document.getElementById('episode-count').textContent = data.episode_count;
                }
                
                if (data.rewards !== undefined) {
                    document.getElementById('rewards').textContent = JSON.stringify(data.rewards);
                }
                
                if (data.terminations !== undefined) {
                    document.getElementById('terminations').textContent = JSON.stringify(data.terminations);
                }
                
                if (!data.running) {
                    clearInterval(framePollingInterval);
                    document.getElementById('start-button').disabled = false;
                    document.getElementById('stop-button').disabled = true;
                    document.getElementById('status').textContent = 'Stopped';
                }
            })
            .catch(error => {
                console.error('Error fetching frame:', error);
                clearInterval(framePollingInterval);
            });
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Render the main page."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/start', methods=['POST'])
def start_simulation():
    """Start the simulation with the given parameters."""
    global env, running, thread, frame_buffer, step_count, episode_count
    
    if running:
        return jsonify({'success': False, 'message': 'Simulation already running'})
    
    # Get parameters from request
    data = request.json
    obstacle_density = data.get('obstacle_density', 0.2)
    min_radius = data.get('min_radius', 0.2)
    max_radius = data.get('max_radius', 0.7)
    num_agents = data.get('num_agents', 2)
    random_seed = data.get('random_seed', 42)
    simulation_speed = data.get('simulation_speed', 10)
    
    # Create environment
    try:
        if env is not None:
            env.close()
        
        env = posggym.make(
            "DrivingContinuousRandom-v0",
            render_mode="rgb_array",
            obstacle_density=obstacle_density,
            obstacle_radius_range=(min_radius, max_radius),
            num_agents=num_agents,
            random_seed=random_seed,
        )
        
        # Reset environment
        env.reset()
        
        # Reset counters
        step_count = 0
        episode_count = 0
        
        # Clear frame buffer
        frame_buffer = []
        
        # Start simulation thread
        running = True
        thread = threading.Thread(target=run_simulation, args=(simulation_speed,))
        thread.daemon = True
        thread.start()
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/stop', methods=['POST'])
def stop_simulation():
    """Stop the simulation."""
    global running
    running = False
    return jsonify({'success': True})

@app.route('/reset', methods=['POST'])
def reset_environment():
    """Reset the environment."""
    global env, step_count, episode_count
    
    if env is not None:
        env.reset()
        step_count = 0
        episode_count = 0
    
    return jsonify({'success': True})

@app.route('/frame')
def get_frame():
    """Get the latest frame from the buffer."""
    global frame_buffer, running, step_count, episode_count
    
    if not frame_buffer:
        return jsonify({
            'frame': None,
            'running': running,
            'step_count': step_count,
            'episode_count': episode_count,
            'rewards': {},
            'terminations': {}
        })
    
    # Get the latest frame data
    frame_data = frame_buffer[-1]
    
    return jsonify({
        'frame': frame_data['frame'],
        'running': running,
        'step_count': step_count,
        'episode_count': episode_count,
        'rewards': frame_data.get('rewards', {}),
        'terminations': frame_data.get('terminations', {})
    })

def run_simulation(simulation_speed):
    """Run the simulation in a separate thread."""
    global env, running, frame_buffer, step_count, episode_count
    
    # Calculate sleep time based on simulation speed
    sleep_time = 1.0 / simulation_speed
    
    while running:
        try:
            # Sample random actions
            actions = {agent: env.action_spaces[agent].sample() for agent in env.agents}
            
            # Step the environment
            step_result = env.step(actions)
            
            # Unpack the step result based on its structure
            if isinstance(step_result, tuple) and len(step_result) == 6:
                # Format: (obs, rewards, terminations, truncations, done, infos)
                next_obs, rewards, terminations, truncations, _, infos = step_result
            else:
                # Standard format: (obs, rewards, terminations, truncations, infos)
                next_obs, rewards, terminations, truncations, infos = step_result
            
            # Increment step count
            step_count += 1
            
            # Render the environment
            frame = env.render()
            
            # Convert frame to base64 encoded PNG
            pil_img = Image.fromarray(frame)
            buff = io.BytesIO()
            pil_img.save(buff, format="PNG")
            frame_base64 = base64.b64encode(buff.getvalue()).decode("utf-8")
            
            # Add frame to buffer (limit buffer size to avoid memory issues)
            frame_buffer.append({
                'frame': frame_base64,
                'rewards': rewards,
                'terminations': terminations
            })
            if len(frame_buffer) > 10:
                frame_buffer = frame_buffer[-10:]
            
            # Check if episode is done
            if all(terminations.values()) or all(truncations.values()):
                env.reset()
                episode_count += 1
            
            # Sleep to control simulation speed
            time.sleep(sleep_time)
        except Exception as e:
            print(f"Error in simulation thread: {e}")
            running = False
            break

def main():
    """Run the web server."""
    parser = argparse.ArgumentParser(description="DrivingContinuousRandom Web Visualizer")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=50452, help="Port to run the server on")
    args = parser.parse_args()
    
    print(f"Starting web server at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)

if __name__ == "__main__":
    main()