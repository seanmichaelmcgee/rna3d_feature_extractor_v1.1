"""
Module for calculating and analyzing RNA pseudo-dihedral angles.

This module calculates various dihedral angles in RNA structures that
characterize the conformation of the RNA backbone and sugar puckers.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import time
import os

def calculate_pseudo_dihedrals(coords_df):
    """
    Calculate pseudo-dihedral angles from C1' atom coordinates.
    
    Args:
        coords_df: DataFrame with C1' atom coordinates
        
    Returns:
        DataFrame with calculated pseudo-dihedral angles
    """
    # Extract coordinates
    if 'x_1' in coords_df.columns:
        x_col, y_col, z_col = 'x_1', 'y_1', 'z_1'
    else:
        # Find coordinate columns
        x_cols = [col for col in coords_df.columns if col.startswith('x_')]
        if not x_cols:
            raise ValueError("No coordinate columns found in the DataFrame")
        
        # Use the first structure
        struct_num = x_cols[0].split('_')[1]
        x_col = f"x_{struct_num}"
        y_col = f"y_{struct_num}"
        z_col = f"z_{struct_num}"
    
    # Extract coordinates as array
    coords = coords_df[[x_col, y_col, z_col]].values
    
    # Calculate pseudo-dihedral angles
    n_residues = len(coords)
    if n_residues < 4:
        raise ValueError("At least 4 residues are required to calculate pseudo-dihedral angles")
    
    # Initialize output DataFrame
    dihedrals = pd.DataFrame({
        'resid': coords_df['resid'].values[1:n_residues-2],
        'eta': np.zeros(n_residues-3),
        'theta': np.zeros(n_residues-3),
        'eta_sin': np.zeros(n_residues-3),
        'eta_cos': np.zeros(n_residues-3),
        'theta_sin': np.zeros(n_residues-3),
        'theta_cos': np.zeros(n_residues-3)
    })
    
    # Calculate dihedral angles for each set of 4 consecutive residues
    for i in range(n_residues - 3):
        # Get four consecutive C1' atoms
        a = coords[i]
        b = coords[i+1]
        c = coords[i+2]
        d = coords[i+3]
        
        # Calculate vectors between points
        ab = b - a
        bc = c - b
        cd = d - c
        
        # Calculate cross products
        cross1 = np.cross(ab, bc)
        cross2 = np.cross(bc, cd)
        
        # Normalize cross products
        norm1 = np.linalg.norm(cross1)
        norm2 = np.linalg.norm(cross2)
        
        if norm1 < 1e-10 or norm2 < 1e-10:
            # Skip if cross product is too small (points are collinear)
            continue
        
        cross1 /= norm1
        cross2 /= norm2
        
        # Calculate dihedral angle (eta)
        cos_eta = np.dot(cross1, cross2)
        # Clamp cos_eta to [-1, 1] to avoid numerical errors
        cos_eta = max(-1.0, min(1.0, cos_eta))
        
        # Calculate sign using triple scalar product
        bc_norm = bc / np.linalg.norm(bc)
        sin_eta = np.dot(np.cross(cross1, cross2), bc_norm)
        
        # Calculate the angle in degrees
        eta = np.degrees(np.arccos(cos_eta))
        if sin_eta < 0:
            eta = -eta
        
        # Calculate theta angle (between consecutive pseudo-dihedral planes)
        if i > 0:
            plane1 = cross1  # Normal to plane defined by a, b, c
            plane2 = cross2  # Normal to plane defined by b, c, d
            
            cos_theta = np.dot(plane1, plane2)
            # Clamp cos_theta to [-1, 1] to avoid numerical errors
            cos_theta = max(-1.0, min(1.0, cos_theta))
            
            theta = np.degrees(np.arccos(cos_theta))
            
            # Store in dataframe
            dihedrals.loc[i-1, 'theta'] = theta
            dihedrals.loc[i-1, 'theta_sin'] = np.sin(np.radians(theta))
            dihedrals.loc[i-1, 'theta_cos'] = np.cos(np.radians(theta))
        
        # Store in dataframe
        dihedrals.loc[i, 'eta'] = eta
        dihedrals.loc[i, 'eta_sin'] = np.sin(np.radians(eta))
        dihedrals.loc[i, 'eta_cos'] = np.cos(np.radians(eta))
    
    return dihedrals

def extract_dihedral_features(coords_df, output_file=None, include_raw_angles=True):
    """
    Extract pseudo-dihedral feature tensors for machine learning.
    
    Args:
        coords_df: DataFrame with coordinates
        output_file: Optional path to save features (.npz or .pkl)
        include_raw_angles: Whether to include raw eta/theta values in addition to sin/cos
        
    Returns:
        NumPy array with dihedral features or a dictionary if include_raw_angles=True
    """
    # Calculate dihedrals
    dihedrals_df = calculate_pseudo_dihedrals(coords_df)
    
    # Convert to feature tensors
    # For each residue, we'll encode the dihedral angles as [sin(η), cos(η), sin(θ), cos(θ)]
    # as required by the Analysis Plan
    
    # Extract the features we need
    n_residues = len(coords_df)
    features = np.zeros((n_residues, 4))  # [sin(η), cos(η), sin(θ), cos(θ)] for each residue
    
    # If raw angles are requested, create arrays for those too
    if include_raw_angles:
        eta_values = np.zeros(n_residues)
        theta_values = np.zeros(n_residues)
        eta_values.fill(np.nan)  # Initialize with NaN
        theta_values.fill(np.nan)
    
    # First, create a mapping from residue IDs to feature indices
    resid_to_index = {resid: i for i, resid in enumerate(coords_df['resid'])}
    
    # Fill in the features for residues that have dihedrals calculated
    for _, row in dihedrals_df.iterrows():
        idx = resid_to_index.get(row['resid'])
        if idx is not None:
            features[idx, 0] = row['eta_sin']
            features[idx, 1] = row['eta_cos']
            features[idx, 2] = row['theta_sin'] if not np.isnan(row['theta_sin']) else 0
            features[idx, 3] = row['theta_cos'] if not np.isnan(row['theta_cos']) else 0
            
            # Store raw angles if requested
            if include_raw_angles:
                eta_values[idx] = row['eta']
                theta_values[idx] = row['theta'] if not np.isnan(row['theta']) else 0
    
    # Create a metadata dictionary with column names
    feature_names = ['eta_sin', 'eta_cos', 'theta_sin', 'theta_cos']
    metadata = {
        'feature_names': feature_names,
        'feature_description': 'Pseudo-dihedral angle features in sin/cos encoding',
        'column_0': 'eta_sin - sine of eta angle',
        'column_1': 'eta_cos - cosine of eta angle',
        'column_2': 'theta_sin - sine of theta angle',
        'column_3': 'theta_cos - cosine of theta angle',
        'extraction_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Optional save to file
    if output_file:
        # Convert Path to string if needed
        output_file_str = str(output_file)
        
        if output_file_str.endswith('.npz'):
            # Save with named keys to include raw angles if requested
            if include_raw_angles:
                np.savez_compressed(output_file, 
                                   features=features,
                                   eta=eta_values,
                                   theta=theta_values,
                                   feature_names=feature_names,
                                   metadata=str(metadata))  # Convert dict to string for numpy storage
            else:
                # Save with feature names
                np.savez_compressed(output_file, 
                                   features=features,
                                   feature_names=feature_names,
                                   metadata=str(metadata))
        elif output_file_str.endswith('.pkl'):
            import pickle
            data_to_save = {
                'features': features,
                'feature_names': feature_names,
                'metadata': metadata
            }
            if include_raw_angles:
                data_to_save.update({
                    'eta': eta_values,
                    'theta': theta_values
                })
            with open(output_file, 'wb') as f:
                pickle.dump(data_to_save, f)
        print(f"Dihedral features saved to {output_file}")
    
    # Return features or dictionary with features and raw angles
    result = {
        'features': features,
        'feature_names': feature_names,
        'metadata': metadata
    }
    
    if include_raw_angles:
        result.update({
            'eta': eta_values,
            'theta': theta_values
        })
        
    return result

def plot_dihedral_traces(dihedrals_df, output_file=None, show_plot=False):
    """
    Plot pseudo-dihedral angle traces.
    
    Args:
        dihedrals_df: DataFrame with dihedral angles
        output_file: Optional path to save plot
        show_plot: Whether to display the plot
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot eta angles
    ax1.plot(dihedrals_df['resid'], dihedrals_df['eta'], 'r-', label='Eta')
    ax1.set_xlabel('Residue ID')
    ax1.set_ylabel('Eta Angle (degrees)')
    ax1.set_title('Pseudo-dihedral Eta Angle vs Residue Position')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot theta angles
    ax2.plot(dihedrals_df['resid'][1:], dihedrals_df['theta'][1:], 'b-', label='Theta')
    ax2.set_xlabel('Residue ID')
    ax2.set_ylabel('Theta Angle (degrees)')
    ax2.set_title('Pseudo-dihedral Theta Angle vs Residue Position')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    # Save plot if requested
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Dihedral angle plot saved to {output_file}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig

def plot_eta_theta_distribution(dihedrals_df, output_file=None, show_plot=False):
    """
    Plot the distribution of eta vs theta angles.
    
    Args:
        dihedrals_df: DataFrame with dihedral angles
        output_file: Optional path to save plot
        show_plot: Whether to display the plot
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Calculate residue indices
    residue_indices = np.arange(len(dihedrals_df))
    
    # Create scatter plot
    scatter = ax.scatter(dihedrals_df['eta'], dihedrals_df['theta'], 
                         c=residue_indices, cmap='viridis', alpha=0.8)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Residue Index')
    
    # Set labels and title
    ax.set_xlabel('Eta Angle (degrees)')
    ax.set_ylabel('Theta Angle (degrees)')
    ax.set_title('Pseudo-dihedral Eta-Theta Distribution')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Save plot if requested
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Eta-Theta distribution plot saved to {output_file}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig
    

def plot_sin_cos_theta_distribution(dihedrals_df, output_file=None, show_plot=False):
    """
    Plot the distribution of sin(theta) vs cos(theta) angles.
    
    Args:
        dihedrals_df: DataFrame with dihedral angles
        output_file: Optional path to save plot
        show_plot: Whether to display the plot
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Calculate residue indices
    residue_indices = np.arange(len(dihedrals_df))
    
    # Create scatter plot
    scatter = ax.scatter(dihedrals_df['theta_sin'], dihedrals_df['theta_cos'], 
                         c=residue_indices, cmap='viridis', alpha=0.8)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Residue Index')
    
    # Set labels and title
    ax.set_xlabel('sin(θ)')
    ax.set_ylabel('cos(θ)')
    ax.set_title('Sin-Cos Encoding of Pseudo-dihedral Theta Angle')
    
    # Add reference unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    x = np.sin(theta)
    y = np.cos(theta)
    ax.plot(x, y, 'k--', alpha=0.3)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Set equal aspect ratio
    ax.set_aspect('equal')
    
    # Set limits to show full unit circle
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    
    # Save plot if requested
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Sin-Cos Theta distribution plot saved to {output_file}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig

if __name__ == "__main__":
    print("RNA Pseudo-dihedral Analysis Module")