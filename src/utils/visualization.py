"""
Visualization Module

This module provides visualization functionality for RNA features,
including RNA structure diagrams, feature matrices, and reports.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import logging
import io
import base64

# Configure logger
logger = logging.getLogger("Visualization")

def visualize_rna_structure(sequence, structure, output_file=None, title=None, figsize=(15, 3)):
    """
    Generate a simple visualization of RNA secondary structure.
    
    Args:
        sequence (str): RNA sequence
        structure (str): Dot-bracket notation of RNA structure
        output_file (str or Path, optional): Path to save the plot. Defaults to None.
        title (str, optional): Plot title. Defaults to None.
        figsize (tuple, optional): Figure size. Defaults to (15, 3).
        
    Returns:
        matplotlib.figure.Figure: Generated figure or None if failed
    """
    try:
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Extract base pairs from structure
        stack = []
        pairs = []
        for i, char in enumerate(structure):
            if char == '(':
                stack.append(i)
            elif char == ')':
                if stack:
                    j = stack.pop()
                    pairs.append((j, i))
        
        # Draw baseline
        ax.plot([0, len(structure)], [0, 0], 'k-', lw=1, alpha=0.5)
        
        # Draw nucleotides
        for i, base in enumerate(sequence):
            ax.text(i, -0.1, base, ha='center', va='top', fontsize=8)
        
        # Draw arcs for paired bases
        for i, j in pairs:
            center = (i + j) / 2
            width = abs(j - i)
            height = width / 5  # Adjust for aesthetics
            ax.add_patch(plt.Rectangle((center - width/2, 0), width, height, 
                      facecolor='none', edgecolor='blue', alpha=0.5))
        
        # Set title
        if title:
            ax.set_title(title)
        else:
            ax.set_title("RNA Secondary Structure")
            
        # Set axis limits and ticks
        ax.set_xlim(-1, len(structure) + 1)
        ax.set_ylim(-0.5, (len(structure)/10) + 1)
        ax.set_yticks([])
        ax.set_xticks(range(0, len(structure), 10))
        
        # Save figure if output file specified
        if output_file:
            plt.savefig(output_file)
            logger.info(f"RNA structure visualization saved to {output_file}")
            
        return fig
        
    except Exception as e:
        logger.error(f"Error visualizing RNA structure: {e}")
        return None
        
def plot_mi_matrix(mi_features, output_file=None, title=None, figsize=(10, 8), cmap='viridis'):
    """
    Plot MI matrix.
    
    Args:
        mi_features (dict): MI features containing scores matrix
        output_file (str or Path, optional): Path to save the plot. Defaults to None.
        title (str, optional): Plot title. Defaults to None.
        figsize (tuple, optional): Figure size. Defaults to (10, 8).
        cmap (str, optional): Colormap. Defaults to 'viridis'.
        
    Returns:
        matplotlib.figure.Figure: Generated figure or None if failed
    """
    try:
        # Extract MI matrix
        if 'scores' in mi_features:
            scores = mi_features['scores']
        elif 'coupling_matrix' in mi_features:
            scores = mi_features['coupling_matrix']
        else:
            logger.error("No scores or coupling matrix found in MI features")
            return None
            
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot matrix
        im = ax.imshow(scores, cmap=cmap, origin='lower')
        plt.colorbar(im, label='Mutual Information')
        
        # Set title
        if title:
            ax.set_title(title)
        else:
            ax.set_title("Mutual Information Matrix")
            
        # Set axis labels
        ax.set_xlabel("Position")
        ax.set_ylabel("Position")
        
        # Save figure if output file specified
        if output_file:
            plt.savefig(output_file)
            logger.info(f"MI matrix plot saved to {output_file}")
            
        return fig
        
    except Exception as e:
        logger.error(f"Error plotting MI matrix: {e}")
        return None
        
def plot_thermodynamic_features(thermo_features, output_file=None, title=None, figsize=(12, 4)):
    """
    Visualize thermodynamic features.
    
    Args:
        thermo_features (dict): Thermodynamic features
        output_file (str or Path, optional): Path to save the plot. Defaults to None.
        title (str, optional): Plot title. Defaults to None.
        figsize (tuple, optional): Figure size. Defaults to (12, 4).
        
    Returns:
        matplotlib.figure.Figure: Generated figure or None if failed
    """
    try:
        # Check for position entropy
        if 'position_entropy' not in thermo_features:
            logger.error("No position entropy found in thermodynamic features")
            return None
            
        entropy = thermo_features['position_entropy']
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot entropy
        ax.plot(range(len(entropy)), entropy, 'b-')
        
        # Set title
        if title:
            ax.set_title(title)
        else:
            ax.set_title("Positional Entropy")
            
        # Set axis labels
        ax.set_xlabel("Position")
        ax.set_ylabel("Entropy")
        ax.grid(alpha=0.3)
        
        # Save figure if output file specified
        if output_file:
            plt.savefig(output_file)
            logger.info(f"Thermodynamic features plot saved to {output_file}")
            
        return fig
        
    except Exception as e:
        logger.error(f"Error plotting thermodynamic features: {e}")
        return None
        
def generate_feature_report(features, target_id=None, output_file=None):
    """
    Create feature report with visualizations.
    
    Args:
        features (dict): Features dictionary with thermodynamic and MI features
        target_id (str, optional): Target ID. Defaults to None.
        output_file (str or Path, optional): Path to save the report. Defaults to None.
        
    Returns:
        str: HTML report or None if failed
    """
    try:
        # Start HTML content
        html = []
        html.append("<!DOCTYPE html>")
        html.append("<html>")
        html.append("<head>")
        html.append("  <title>RNA Feature Report</title>")
        html.append("  <style>")
        html.append("    body { font-family: Arial, sans-serif; margin: 20px; }")
        html.append("    h1, h2 { color: #2c3e50; }")
        html.append("    .section { margin: 20px 0; }")
        html.append("    .feature-table { border-collapse: collapse; width: 100%; }")
        html.append("    .feature-table th, .feature-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
        html.append("    .feature-table th { background-color: #f2f2f2; }")
        html.append("    .plot-container { margin: 20px 0; }")
        html.append("  </style>")
        html.append("</head>")
        html.append("<body>")
        
        # Add header
        if target_id:
            html.append(f"<h1>RNA Feature Report: {target_id}</h1>")
        else:
            html.append("<h1>RNA Feature Report</h1>")
            
        # Add thermodynamic features section if available
        if 'thermo' in features and features['thermo']:
            thermo = features['thermo']
            html.append("<h2>Thermodynamic Features</h2>")
            html.append("<div class='section'>")
            
            # Add basic feature table
            html.append("<table class='feature-table'>")
            html.append("  <tr><th>Feature</th><th>Value</th></tr>")
            
            for key in ['mfe', 'ensemble_energy', 'prob_of_mfe', 'mean_entropy']:
                if key in thermo:
                    html.append(f"  <tr><td>{key}</td><td>{thermo[key]}</td></tr>")
                    
            html.append("</table>")
            
            # Add structure visualization if available
            if 'structure' in thermo and 'sequence' in thermo:
                html.append("<h3>RNA Structure</h3>")
                html.append("<div class='plot-container'>")
                
                # Generate structure plot
                fig = visualize_rna_structure(
                    thermo['sequence'], 
                    thermo['structure'],
                    title=f"RNA Structure"
                )
                
                if fig:
                    # Convert plot to base64 string
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png')
                    buf.seek(0)
                    img_str = base64.b64encode(buf.read()).decode('utf-8')
                    
                    # Add image to HTML
                    html.append(f"<img src='data:image/png;base64,{img_str}' alt='RNA Structure' />")
                    plt.close(fig)
                    
                html.append("</div>")
                
            # Add entropy plot if available
            if 'position_entropy' in thermo:
                html.append("<h3>Positional Entropy</h3>")
                html.append("<div class='plot-container'>")
                
                # Generate entropy plot
                fig = plot_thermodynamic_features(
                    thermo,
                    title=f"Positional Entropy"
                )
                
                if fig:
                    # Convert plot to base64 string
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png')
                    buf.seek(0)
                    img_str = base64.b64encode(buf.read()).decode('utf-8')
                    
                    # Add image to HTML
                    html.append(f"<img src='data:image/png;base64,{img_str}' alt='Positional Entropy' />")
                    plt.close(fig)
                    
                html.append("</div>")
                
            html.append("</div>")
            
        # Add MI features section if available
        if 'mi' in features and features['mi']:
            mi = features['mi']
            html.append("<h2>Mutual Information Features</h2>")
            html.append("<div class='section'>")
            
            # Add MI info
            if 'method' in mi:
                html.append(f"<p>Method: {mi['method']}</p>")
                
            if 'single_sequence' in mi and mi['single_sequence']:
                html.append("<p><strong>Note:</strong> Single-sequence MSA detected, optimized calculation used.</p>")
                
            # Add top pairs if available
            if 'top_pairs' in mi and len(mi['top_pairs']) > 0:
                html.append("<h3>Top MI Pairs</h3>")
                html.append("<table class='feature-table'>")
                html.append("  <tr><th>Position 1</th><th>Position 2</th><th>Score</th></tr>")
                
                for i in range(min(5, len(mi['top_pairs']))):
                    pair = mi['top_pairs'][i]
                    html.append(f"  <tr><td>{pair[0]}</td><td>{pair[1]}</td><td>{pair[2]:.4f}</td></tr>")
                    
                html.append("</table>")
                
            # Add MI matrix plot if available
            if 'scores' in mi or 'coupling_matrix' in mi:
                html.append("<h3>Mutual Information Matrix</h3>")
                html.append("<div class='plot-container'>")
                
                # Generate MI matrix plot
                fig = plot_mi_matrix(
                    mi,
                    title=f"Mutual Information Matrix"
                )
                
                if fig:
                    # Convert plot to base64 string
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png')
                    buf.seek(0)
                    img_str = base64.b64encode(buf.read()).decode('utf-8')
                    
                    # Add image to HTML
                    html.append(f"<img src='data:image/png;base64,{img_str}' alt='MI Matrix' />")
                    plt.close(fig)
                    
                html.append("</div>")
                
            html.append("</div>")
            
        # Close HTML
        html.append("</body>")
        html.append("</html>")
        
        # Save HTML to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write("\n".join(html))
            logger.info(f"Feature report saved to {output_file}")
            
        return "\n".join(html)
        
    except Exception as e:
        logger.error(f"Error generating feature report: {e}")
        return None