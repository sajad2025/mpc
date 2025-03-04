#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from core_solver import EgoConfig, SimConfig, generate_controls
from plots import plot_results_xy

def create_docs_dir():
    """Get the path to the docs directory."""
    # Get the project root directory (parent of the src directory)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    docs_dir = os.path.join(project_root, 'docs')
    
    # The docs directory already exists, so just return the path
    return docs_dir

def run_scenario(scenario):
    """
    Run a single scenario with the given parameters.
    
    Args:
        scenario: Dictionary containing scenario parameters
    
    Returns:
        results: Dictionary with simulation results or None if failed
        ego: EgoConfig object used for this scenario
        success: Boolean indicating whether the scenario was solved successfully
    """
    print(f"Running {scenario['name']} scenario...")
    
    # Create configurations
    ego = EgoConfig()
    sim_cfg = SimConfig()
    
    # Apply scenario-specific parameters
    ego.corridor_width = scenario.get('corridor_width', 2.0)
    sim_cfg.duration = scenario.get('duration', 10.0)
    
    # Set start and goal states
    ego.state_start = scenario['start_state']
    ego.state_final = scenario['goal_state']
    ego.velocity_min = scenario['velocity_min']
    
    # Generate controls
    results = generate_controls(ego, sim_cfg)
    
    # Check if solver succeeded
    if results is None:
        print(f"Failed to solve {scenario['name']} scenario")
        return None, ego, False
    
    # Plot and save results
    docs_dir = create_docs_dir()
    save_path = os.path.join(docs_dir, f"{scenario['name'].lower().replace(' ', '_')}.png")
    
    fig, ax = plot_results_xy(
        results, 
        ego, 
        title=f"{scenario['name']} Scenario", 
        show_arrows=True, 
        save_path=save_path
    )
    
    plt.close(fig)  # Close figure to avoid memory issues when running many scenarios
    
    return results, ego, True

def run_sequential_planning(segments, name="Sequential_Path"):
    """
    Run sequential path planning through a series of waypoints.
    
    Args:
        segments: List of scenario dictionaries defining each path segment
        name: Name for the overall sequential path
    
    Returns:
        combined_results: Dictionary containing the combined path data
        success: Boolean indicating if all segments were solved successfully
    """
    print(f"\n{'='*50}")
    print(f"SEQUENTIAL PATH PLANNING: {name}")
    print(f"{'='*50}")
    
    # Variables to store combined results
    combined_trajectory = []
    combined_time = []
    combined_controls = []
    all_segment_results = []
    current_time = 0
    
    # Run each segment sequentially
    docs_dir = create_docs_dir()
    success = True
    
    for i, segment in enumerate(segments):
        print(f"\nSegment {i+1}: {segment['name']} - From {segment['start_state'][:3]} to {segment['goal_state'][:3]}")
        
        # Run the segment
        results, ego, segment_success = run_scenario(segment)
        
        if not segment_success:
            print(f"Failed to solve segment {i+1}. Aborting sequential planning.")
            success = False
            break
        
        # Store segment results
        all_segment_results.append({
            'segment': i+1,
            'name': segment['name'],
            'results': results,
            'ego': ego
        })
        
        # Adjust time to be continuous from previous segments
        segment_time = results['t'].copy()
        if i > 0:
            segment_time += current_time
        
        # Update current_time for next segment
        current_time = segment_time[-1]
        
        # Combine results (excluding the duplicate points at segment transitions)
        if i == 0:
            combined_trajectory.append(results['x'])
            combined_time.append(segment_time)
            combined_controls.append(results['u'])
        else:
            # Skip the first point of subsequent segments to avoid duplicates
            combined_trajectory.append(results['x'][1:])
            combined_time.append(segment_time[1:])
            combined_controls.append(results['u'][1:])
    
    if not success:
        return None, False
    
    # Create combined arrays (convert list of arrays to single array)
    combined_x = np.vstack([segment for segment in combined_trajectory])
    combined_t = np.concatenate(combined_time)
    combined_u = np.vstack([segment for segment in combined_controls])
    
    # Create combined results dictionary
    combined_results = {
        't': combined_t,
        'x': combined_x,
        'u': combined_u,
        'status': 0,  # All segments successful
        'segments': all_segment_results
    }
    
    # Create a figure for the full path
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot the full trajectory
    ax.plot(combined_x[:, 0], combined_x[:, 1], 'b-', linewidth=2, label='Complete path')
    
    # Plot waypoints
    for i, segment in enumerate(segments):
        if i == 0:
            # First waypoint (start)
            ax.plot(segment['start_state'][0], segment['start_state'][1], 'go', markersize=10, label='Start')
            # Add text label for the start point
            ax.text(segment['start_state'][0]+0.5, segment['start_state'][1]+0.5, 
                   f"A({segment['start_state'][0]}, {segment['start_state'][1]}, {segment['start_state'][2]:.2f})",
                   fontsize=10)
            
        # End of each segment (waypoints)
        ax.plot(segment['goal_state'][0], segment['goal_state'][1], 'ro', markersize=8)
        
        # Get waypoint letter based on index (A-F)
        waypoint_letter = chr(ord('A') + i + 1)  # B, C, D, E, F for segments 0-4
        
        # Add text label for each waypoint
        ax.text(segment['goal_state'][0]+0.5, segment['goal_state'][1]+0.5, 
               f"{waypoint_letter}({segment['goal_state'][0]}, {segment['goal_state'][1]}, {segment['goal_state'][2]:.2f})",
               fontsize=10)
    
    # Add direction arrows along the path
    arrow_interval = max(1, len(combined_x) // 40)  # Adjust for appropriate number of arrows
    sampled_points = combined_x[::arrow_interval]
    directions_x = np.cos(sampled_points[:, 2])
    directions_y = np.sin(sampled_points[:, 2])
    ax.quiver(sampled_points[:, 0], sampled_points[:, 1], 
             directions_x, directions_y,
             color='blue', alpha=0.3, scale=12, scale_units='inches',
             width=0.003, headwidth=4, headlength=5)
    
    # Set plot properties
    ax.grid(True)
    ax.set_aspect('equal')
    ax.set_xlabel('X position (m)')
    ax.set_ylabel('Y position (m)')
    ax.set_title(f"Sequential Path Planning: {name}")
    ax.legend()
    
    # Save the figure
    save_path = os.path.join(docs_dir, f"{name.lower().replace(' ', '_')}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"\nSequential planning complete. Results saved to: {save_path}")
    
    return combined_results, success

# Define scenarios list - easy to add more scenarios later
scenarios = [
    # {
    #     'name': 'U-Turn',
    #     'corridor_width': 7.0,
    #     'duration': 40.0,
    #     'velocity_min': 0.0,
    #     'start_state': [0, 0, np.pi/2, 1.0, 0],  # x, y, theta (east), velocity, steering
    #     'goal_state': [10, 0, -np.pi/2, 1.0, 0],  # x, y, theta (west), velocity, steering
    # },
    # {
    #     'name': 'Corner Turn',
    #     'corridor_width': 3.0,
    #     'duration': 20.0,
    #     'velocity_min': 0.0,
    #     'start_state': [0, 0, np.pi/2, 1.0, 0],  # x, y, theta (east), velocity, steering
    #     'goal_state': [7, 7, 0, 1.0, 0],  # x, y, theta (north), velocity, steering
    # },
    # {
    #     'name': 'line_change',
    #     'corridor_width': 7.0,
    #     'duration': 22.0,
    #     'velocity_min': 0.0,
    #     'start_state': [0, 0, 0,  2.0, 0],  # x, y, theta (east), velocity, steering
    #     'goal_state': [30, -5, 0, 1.0, 0],  # x, y, theta (west), velocity, steering
    # },
    {
        'name': 'line',
        'corridor_width': 3.0,
        'duration': 15.0,
        'velocity_min': 0.0,
        'start_state': [0, 0, 0,  1.0, 0],  # x, y, theta (east), velocity, steering
        'goal_state': [30, 0, 0,  2.0, 0],  # x, y, theta (west), velocity, steering
    },
]

# Define waypoints for sequential planning (A through F based on the diagram)
sequential_segments = [
    {
        'name': 'A_to_B',
        'corridor_width': 7.0,
        'duration': 22.0,
        'velocity_min': 0.0,
        'start_state': [30, 35, np.pi, 2.0, 0],  # x, y, theta (west), velocity, steering
        'goal_state': [0, 30, np.pi, 1.0, 0],    # x, y, theta (west), velocity, steering
        'description': 'From A to B: Moving west along the top'
    },
    {
        'name': 'B_to_C',
        'corridor_width': 7.0,
        'duration': 40.0,
        'velocity_min': 0.0,
        'start_state': [0, 30, np.pi, 1, 0],   # x, y, theta (west), velocity, steering
        'goal_state': [0, 20, 0, 1, 0],        # x, y, theta (east), velocity, steering
        'description': 'From B to C: Moving south slightly and turning east'
    },
    {
        'name': 'C_to_D',
        'corridor_width': 7.0,
        'duration': 22.0,
        'velocity_min': 0.0,
        'start_state': [0, 20, 0, 1.0, 0],       # x, y, theta (east), velocity, steering
        'goal_state': [30, 15, 0, 1.0, 0],       # x, y, theta (east), velocity, steering
        'description': 'From C to D: Moving east and slightly down'
    },
    {
        'name': 'D_to_E',
        'corridor_width': 3.0,
        'duration': 20.0,
        'velocity_min': 0.0,
        'start_state': [30, 15, 0, 1.0, 0],      # x, y, theta (east), velocity, steering
        'goal_state': [37, 8, -np.pi/2, 1.0, 0], # x, y, theta (south), velocity, steering
        'description': 'From D to E: Moving east-south-east and turning south'
    },
    {
        'name': 'E_to_F',
        'corridor_width': 3.0,
        'duration': 15.0,
        'velocity_min': 0.0,
        'start_state': [37, 8, -np.pi/2, 1.0, 0], # x, y, theta (south), velocity, steering
        'goal_state': [37, -22, -np.pi/2, 3.0, 0],   # x, y, theta (south), velocity, steering
        'description': 'From E to F: Moving straight south'
    }
]

if __name__ == "__main__":
    # Create docs directory
    docs_dir = create_docs_dir()
    print(f"Results will be saved to: {docs_dir}")
    
    # Choose whether to run individual scenarios or sequential planning
    run_individual_scenarios = False
    run_sequential_path = True
    
    # Run all individual scenarios if selected
    if run_individual_scenarios:
        results_by_scenario = {}
        results_all = []  # List to store all successful results
        success_count = 0
        failed_count = 0
        
        for scenario in scenarios:
            print(f"\n{'='*50}")
            print(f"SCENARIO: {scenario['name']}")
            print(f"Description: {scenario.get('description', 'No description')}")
            print(f"{'='*50}")
            
            # Run the scenario
            results, ego, success = run_scenario(scenario)
            
            if success:
                # Store results
                results_by_scenario[scenario['name']] = {
                    'results': results,
                    'ego': ego,
                    'scenario': scenario
                }
                
                # Append to results_all
                results_all.append({
                    'name': scenario['name'],
                    'trajectory': results['x'],
                    'time': results['t'],
                    'controls': results['u'],
                    'status': results['status'],
                    'start_state': scenario['start_state'],
                    'goal_state': scenario['goal_state'],
                    'ego_config': ego
                })
                
                print(f"Completed {scenario['name']} scenario successfully")
                print(f"Results saved to: {os.path.join(docs_dir, f"{scenario['name'].lower().replace(' ', '_')}.png")}")
                success_count += 1
            else:
                print(f"Failed to solve {scenario['name']} scenario")
                failed_count += 1
        
        print(f"\nAll scenarios completed. Success: {success_count}, Failed: {failed_count}")
        print(f"Total successful trajectories in results_all: {len(results_all)}")
        
        if failed_count > 0:
            print(f"WARNING: {failed_count} scenarios failed to solve!")
            
        # Save all results to a numpy file
        if len(results_all) > 0:
            results_file = os.path.join(docs_dir, "all_results.npy")
            np.save(results_file, results_all)
            print(f"All results saved to: {results_file}")
    
    # Run sequential path planning if selected
    if run_sequential_path:
        # Run the sequential path planning with all waypoints
        combined_results, success = run_sequential_planning(
            sequential_segments, 
            name="A_to_F_Path"
        )
        
        if success:
            # Save the combined results
            sequential_file = os.path.join(docs_dir, "sequential_path_results.npy")
            np.save(sequential_file, combined_results)
            print(f"Sequential planning results saved to: {sequential_file}")
            
            # Also save in the format expected by results_all
            results_all = [{
                'name': "Sequential_A_to_F",
                'trajectory': combined_results['x'],
                'time': combined_results['t'],
                'controls': combined_results['u'],
                'status': combined_results['status'],
                'start_state': sequential_segments[0]['start_state'],
                'goal_state': sequential_segments[-1]['goal_state'],
                'segments': combined_results['segments']
            }]
            
            # Save to a separate file to avoid overwriting individual scenario results
            all_sequential_file = os.path.join(docs_dir, "sequential_all_results.npy")
            np.save(all_sequential_file, results_all)
        else:
            print("Sequential path planning failed.") 