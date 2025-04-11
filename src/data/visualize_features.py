#!/usr/bin/env python3
"""
RNA Feature Visualization Tool

This script provides a command-line interface to visualize RNA features
stored in NPZ files generated by the extract_features_simple.py tool.

It creates visualization plots for all features in the NPZ files.
"""

import os
import sys
import argparse
import glob
from pathlib import Path
import traceback

# Import visualization libraries
try:
    import numpy as np
    has_numpy = True
except ImportError:
    has_numpy = False
    print("ERROR: NumPy is required for this tool")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    has_matplotlib = True
except ImportError:
    has_matplotlib = False
    print("ERROR: Matplotlib is required for visualization")
    sys.exit(1)

def visualize_features(npz_file, output_dir=None, show_plots=False):
    """
    Visualize RNA features stored in an NPZ file.
    
    Parameters:
    -----------
    npz_file : str or Path
        Path to the NPZ file containing RNA features
    output_dir : str or Path, optional
        Directory to save visualization files. If None, uses the same directory as NPZ file
    show_plots : bool
        Whether to display plots interactively
        
    Returns:
    --------
    dict
        Paths to generated visualization files
    """
    # Load NPZ file
    try:
        features = np.load(npz_file)
        npz_path = Path(npz_file)
        
        # Extract sequence ID from filename
        seq_id = npz_path.stem.replace('_features', '')
        
        # Determine output directory
        if output_dir is None:
            # If no output dir specified, use same directory as NPZ file
            vis_dir = npz_path.parent / f"{seq_id}_visualizations"
        else:
            # If output dir specified, put in subdirectory with sequence ID
            vis_dir = Path(output_dir) / seq_id
        
        # Create output directory
        vis_dir.mkdir(exist_ok=True, parents=True)
        
        # Check required features
        required_features = ['sequence', 'structure', 'position_entropy', 'accessibility', 'pairing_probs']
        missing_features = [f for f in required_features if f not in features]
        if missing_features:
            print(f"WARNING: Missing required features: {', '.join(missing_features)}")
            return {}
        
        # Extract features
        sequence = str(features['sequence'])
        structure = str(features['structure'])
        position_entropy = features['position_entropy']
        accessibility = features['accessibility']
        pairing_probs = features['pairing_probs']
        
        # Visualizations dictionary to track generated files
        visualization_files = {}
        
        # 1. Base pair probability matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(pairing_probs, cmap='viridis', origin='lower')
        plt.colorbar(label='Base Pair Probability')
        plt.title(f"{seq_id}: Base Pair Probability Matrix")
        plt.xlabel('Nucleotide Position')
        plt.ylabel('Nucleotide Position')
        
        # Save the plot
        bpp_file = vis_dir / f"{seq_id}_bpp_matrix.png"
        plt.savefig(bpp_file, dpi=150, bbox_inches='tight')
        visualization_files['bpp_matrix'] = str(bpp_file)
        if not show_plots:
            plt.close()
        
        # 2. Position-wise entropy
        plt.figure(figsize=(12, 4))
        plt.plot(range(1, len(position_entropy) + 1), position_entropy, 'b-', linewidth=2)
        plt.fill_between(range(1, len(position_entropy) + 1), position_entropy, alpha=0.3)
        plt.title(f"{seq_id}: Positional Entropy")
        plt.xlabel('Nucleotide Position')
        plt.ylabel('Shannon Entropy')
        plt.grid(True, alpha=0.3)
        
        # Add secondary structure annotation at top
        max_entropy = max(position_entropy) if len(position_entropy) > 0 else 1.0
        for i, char in enumerate(structure):
            color = 'gray'
            if char == '(':
                color = 'green'
            elif char == ')':
                color = 'red'
            plt.text(i+1, max_entropy*1.1, char, color=color, ha='center', fontsize=8)
        
        # Add sequence annotation
        for i, char in enumerate(sequence):
            plt.text(i+1, max_entropy*1.05, char, ha='center', fontsize=8)
            
        # Adjust y-axis to make room for annotations
        try:
            plt.ylim(0, max_entropy*1.2)
        except (ValueError, TypeError):
            # Handle case where entropy might be all zeros
            plt.ylim(0, 1.0)
        
        # Save the plot
        entropy_file = vis_dir / f"{seq_id}_positional_entropy.png"
        plt.savefig(entropy_file, dpi=150, bbox_inches='tight')
        visualization_files['positional_entropy'] = str(entropy_file)
        if not show_plots:
            plt.close()
        
        # 3. Accessibility plot
        plt.figure(figsize=(12, 4))
        plt.plot(range(1, len(accessibility) + 1), accessibility, 'r-', linewidth=2)
        plt.fill_between(range(1, len(accessibility) + 1), accessibility, alpha=0.3)
        plt.title(f"{seq_id}: Nucleotide Accessibility")
        plt.xlabel('Nucleotide Position')
        plt.ylabel('Accessibility')
        plt.ylim(0, 1.05)
        plt.grid(True, alpha=0.3)
        
        # Add secondary structure annotation
        for i, char in enumerate(structure):
            color = 'gray'
            if char == '(':
                color = 'green'
            elif char == ')':
                color = 'red'
            plt.text(i+1, 1.03, char, color=color, ha='center', fontsize=8)
        
        # Save the plot
        access_file = vis_dir / f"{seq_id}_accessibility.png"
        plt.savefig(access_file, dpi=150, bbox_inches='tight')
        visualization_files['accessibility'] = str(access_file)
        if not show_plots:
            plt.close()
        
        # 4. Arc diagram of MFE structure
        try:
            fig, ax = plt.subplots(figsize=(12, 4))
            
            # Draw the backbone
            seq_len = len(sequence)
            x = np.arange(1, seq_len + 1)
            y = np.zeros_like(x, dtype=float)
            ax.plot(x, y, 'k-', lw=1, alpha=0.3)
            
            # Add nucleotide labels
            for i, base in enumerate(sequence):
                ax.text(i+1, -0.02, base, ha='center', va='top', fontsize=8)
            
            # Parse structure to get pairs
            stack = []
            pairs = []
            for i, char in enumerate(structure):
                if char == '(':
                    stack.append(i)
                elif char == ')':
                    if stack:
                        j = stack.pop()
                        pairs.append((j, i))
            
            # Draw arcs for base pairs with color gradient by probability
            max_height = 0.5
            for i, j in pairs:
                # Arc height is proportional to distance between paired bases
                dist = j - i
                height = 0.1 + (dist / seq_len) * max_height
                
                # Get the base pair probability for color gradient
                prob = pairing_probs[i, j]
                
                # Create arc
                theta = np.linspace(0, np.pi, 50)
                x_arc = np.linspace(i+1, j+1, 50)
                y_arc = height * np.sin(theta)
                
                # Color by probability (handle both old and new matplotlib versions)
                try:
                    # New way (matplotlib >= 3.7)
                    import matplotlib.colormaps as colormaps
                    color = colormaps['viridis'](prob)
                except (ImportError, AttributeError):
                    # Fallback to old way
                    color = cm.get_cmap('viridis')(prob)
                
                ax.plot(x_arc, y_arc, '-', lw=1.5, color=color, alpha=0.8)
            
            # Add title and labels
            ax.set_title(f"{seq_id}: Minimum Free Energy Structure")
            ax.set_xlabel('Nucleotide Position')
            ax.set_yticks([])
            
            # Set axis limits
            ax.set_xlim(0, seq_len + 1)
            ax.set_ylim(-0.05, max_height + 0.1)
            
            # Save the figure
            mfe_file = vis_dir / f"{seq_id}_mfe_structure.png"
            plt.savefig(mfe_file, dpi=150, bbox_inches='tight')
            visualization_files['mfe_structure'] = str(mfe_file)
            if not show_plots:
                plt.close()
                
        except Exception as e:
            print(f"Error generating MFE structure visualization: {e}")
        
        # 5. Feature summary for scalar values
        feature_keys = [k for k in features.keys() if isinstance(features[k], (int, float, np.integer, np.floating)) 
                      and k not in ['length']]
        
        if feature_keys:
            # Create a summary of scalar features
            plt.figure(figsize=(10, 6))
            
            # Extract feature values
            feature_values = [float(features[k]) for k in feature_keys]
            
            # Normalize values for color mapping
            norm = Normalize(vmin=min(feature_values), vmax=max(feature_values))
            colors = plt.cm.viridis(norm(feature_values))
            
            # Plot as horizontal bars
            y_pos = np.arange(len(feature_keys))
            plt.barh(y_pos, feature_values, color=colors, alpha=0.7)
            
            # Add feature names and values
            for i, (name, value) in enumerate(zip(feature_keys, feature_values)):
                plt.text(max(0, min(value, min(feature_values))), i, f" {name}: {value:.4f}", 
                        ha='left', va='center', fontsize=9)
            
            # Add labels and title
            plt.yticks(y_pos, feature_keys)
            plt.title(f"{seq_id}: Scalar Feature Summary")
            plt.xlabel('Value')
            plt.tight_layout()
            
            # Save the plot
            summary_file = vis_dir / f"{seq_id}_scalar_features.png"
            plt.savefig(summary_file, dpi=150, bbox_inches='tight')
            visualization_files['scalar_features'] = str(summary_file)
            if not show_plots:
                plt.close()
        
        # 6. GC content visualization (if available)
        if 'gc_content' in features and 'sequence' in features:
            plt.figure(figsize=(10, 4))
            
            # Count GC at each position
            gc_mask = np.array([base in 'GC' for base in sequence], dtype=int)
            window_size = min(15, len(sequence))
            
            # Plot position-wise GC content (1 for G/C, 0 for A/U/T)
            plt.scatter(range(1, len(gc_mask) + 1), gc_mask, 
                      c=['g' if x > 0 else 'b' for x in gc_mask], 
                      alpha=0.5, s=30)
            
            # Calculate and plot rolling GC content if sequence is long enough
            if len(sequence) > window_size*2:
                try:
                    # Use a proper rolling window to avoid dimension mismatch
                    rolling_gc = []
                    for i in range(len(gc_mask) - window_size + 1):
                        rolling_gc.append(np.mean(gc_mask[i:i+window_size]))
                    
                    # Plot the rolling average - ensure x and y have same dimensions
                    x_values = np.arange(1, len(rolling_gc) + 1)
                    plt.plot(x_values, rolling_gc, 'g-', linewidth=2, 
                           label=f'GC content (window size: {window_size})')
                except Exception as e:
                    print(f"Error calculating rolling GC content: {e}")
            
            # Add horizontal line for overall GC content
            plt.axhline(y=features['gc_content'], color='r', linestyle='--', 
                      label=f'Overall GC: {features["gc_content"]:.2f}')
            
            plt.title(f"{seq_id}: GC Content Analysis")
            plt.xlabel('Nucleotide Position')
            plt.ylabel('GC Content')
            plt.ylim(-0.05, 1.05)
            plt.legend(loc='best')
            plt.grid(True, alpha=0.3)
            
            # Save the plot
            gc_file = vis_dir / f"{seq_id}_gc_content.png"
            plt.savefig(gc_file, dpi=150, bbox_inches='tight')
            visualization_files['gc_content'] = str(gc_file)
            if not show_plots:
                plt.close()
        
        # 7. Pairing vs Unpairing visualization
        if all(k in features for k in ['paired_count', 'unpaired_count', 'structure']):
            plt.figure(figsize=(10, 5))
            
            # Create a pie chart for paired vs unpaired
            paired = features['paired_count']
            unpaired = features['unpaired_count']
            
            # Calculate percentages
            total = paired + unpaired
            paired_pct = paired / total * 100 if total > 0 else 0
            unpaired_pct = unpaired / total * 100 if total > 0 else 0
            
            # Plot pie chart
            plt.subplot(1, 2, 1)
            plt.pie([paired_pct, unpaired_pct], 
                   labels=[f'Paired ({paired_pct:.1f}%)', f'Unpaired ({unpaired_pct:.1f}%)'],
                   colors=['#1f77b4', '#ff7f0e'],
                   autopct='%1.1f%%',
                   startangle=90,
                   explode=(0.05, 0))
            plt.title('Base Pairing Distribution')
            
            # Plot structure composition on right side
            plt.subplot(1, 2, 2)
            
            # Count structure elements
            structure_counts = {
                'stem': structure.count('('),  # Opening brackets represent stems
                'loop': structure.count('.')   # Dots represent unpaired regions
            }
            plt.bar(structure_counts.keys(), structure_counts.values(), color=['green', 'orange'])
            
            # Add counts on bars
            for i, (key, value) in enumerate(structure_counts.items()):
                plt.text(i, value + 0.5, str(value), ha='center')
                
            plt.title('Structure Composition')
            plt.tight_layout()
            
            # Save the plot
            pairing_file = vis_dir / f"{seq_id}_pairing_analysis.png"
            plt.savefig(pairing_file, dpi=150, bbox_inches='tight')
            visualization_files['pairing_analysis'] = str(pairing_file)
            if not show_plots:
                plt.close()
                
        # 8. Energy analysis
        if all(k in features for k in ['mfe', 'ensemble_energy', 'prob_of_mfe']):
            plt.figure(figsize=(8, 6))
            
            # Create a bar chart of energy values
            energy_data = {
                'MFE': features['mfe'],
                'Ensemble Energy': features['ensemble_energy']
            }
            
            # Plot energies
            plt.subplot(2, 1, 1)
            colors = ['#2ca02c', '#d62728'] 
            plt.bar(energy_data.keys(), energy_data.values(), color=colors)
            
            # Add values on bars
            for i, (key, value) in enumerate(energy_data.items()):
                plt.text(i, value - 0.5 if value < 0 else value + 0.5, 
                       f"{value:.2f}", ha='center', 
                       color='white' if value < 0 else 'black')
                
            plt.title('Energy Analysis')
            plt.ylabel('Energy (kcal/mol)')
            plt.grid(axis='y', alpha=0.3)
            
            # Plot MFE probability as a gauge
            plt.subplot(2, 1, 2)
            mfe_prob = features['prob_of_mfe']
            
            # Create a horizontal gauge
            plt.barh(['MFE Probability'], [mfe_prob], color='#1f77b4')
            plt.barh(['MFE Probability'], [1 - mfe_prob], left=[mfe_prob], color='#d3d3d3')
            
            # Add percentage text
            plt.text(mfe_prob/2, 0, f"{mfe_prob*100:.1f}%", 
                   ha='center', va='center', color='white' if mfe_prob > 0.3 else 'black')
            
            plt.xlim(0, 1)
            plt.title('Probability of MFE Structure')
            plt.tight_layout()
            
            # Save the plot
            energy_file = vis_dir / f"{seq_id}_energy_analysis.png"
            plt.savefig(energy_file, dpi=150, bbox_inches='tight')
            visualization_files['energy_analysis'] = str(energy_file)
            if not show_plots:
                plt.close()
                
        # 9. Combine structure, entropy and accessibility in one plot
        if all(k in features for k in ['structure', 'position_entropy', 'accessibility']):
            # Create a figure with three subplots
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
            
            # Plot 1: Structure as an arc diagram
            ax1.set_title(f"{seq_id}: Structure, Entropy and Accessibility")
            
            # Draw the backbone
            x = np.arange(1, len(sequence) + 1)
            y = np.zeros_like(x, dtype=float)
            ax1.plot(x, y, 'k-', lw=1, alpha=0.3)
            
            # Add nucleotide labels
            for i, base in enumerate(sequence):
                ax1.text(i+1, -0.02, base, ha='center', va='top', fontsize=8)
            
            # Parse structure to get pairs
            stack = []
            pairs = []
            for i, char in enumerate(structure):
                if char == '(':
                    stack.append(i)
                elif char == ')':
                    if stack:
                        j = stack.pop()
                        pairs.append((j, i))
            
            # Draw arcs for base pairs
            max_height = 0.5
            for i, j in pairs:
                # Arc height is proportional to distance between paired bases
                dist = j - i
                height = 0.1 + (dist / len(sequence)) * max_height
                
                # Get the base pair probability for color gradient
                prob = pairing_probs[i, j]
                
                # Create arc
                theta = np.linspace(0, np.pi, 50)
                x_arc = np.linspace(i+1, j+1, 50)
                y_arc = height * np.sin(theta)
                
                # Color by probability (handle both old and new matplotlib versions)
                try:
                    # New way (matplotlib >= 3.7)
                    import matplotlib.colormaps as colormaps
                    color = colormaps['viridis'](prob)
                except (ImportError, AttributeError):
                    # Fallback to old way
                    color = cm.get_cmap('viridis')(prob)
                
                ax1.plot(x_arc, y_arc, '-', lw=1.5, color=color, alpha=0.8)
            
            ax1.set_ylim(-0.05, max_height + 0.1)
            ax1.set_ylabel('Structure')
            ax1.set_yticks([])
            
            # Plot 2: Position entropy
            ax2.plot(range(1, len(position_entropy) + 1), position_entropy, 'b-', linewidth=2)
            ax2.fill_between(range(1, len(position_entropy) + 1), position_entropy, alpha=0.3)
            ax2.set_ylabel('Shannon Entropy')
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Accessibility
            ax3.plot(range(1, len(accessibility) + 1), accessibility, 'r-', linewidth=2)
            ax3.fill_between(range(1, len(accessibility) + 1), accessibility, alpha=0.3)
            ax3.set_ylim(0, 1.05)
            ax3.set_ylabel('Accessibility')
            ax3.set_xlabel('Nucleotide Position')
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save the plot
            combined_file = vis_dir / f"{seq_id}_combined_analysis.png"
            plt.savefig(combined_file, dpi=150, bbox_inches='tight')
            visualization_files['combined_analysis'] = str(combined_file)
            if not show_plots:
                plt.close()
        
        print(f"Generated {len(visualization_files)} visualization files in {vis_dir}")
        return visualization_files
        
    except Exception as e:
        print(f"Error visualizing {npz_file}: {e}")
        traceback.print_exc()
        return {}

def batch_visualize_features(npz_files, output_dir=None, show_plots=False):
    """
    Process multiple NPZ files in batch mode.
    
    Parameters:
    -----------
    npz_files : list
        List of NPZ file paths
    output_dir : str or Path, optional
        Directory to save visualization files
    show_plots : bool
        Whether to display plots interactively
        
    Returns:
    --------
    dict
        Summary of visualization results
    """
    results = {}
    success_count = 0
    error_count = 0
    vis_count = 0
    
    for npz_file in npz_files:
        print(f"Visualizing {npz_file}")
        
        # Process the file
        vis_files = visualize_features(
            npz_file=npz_file,
            output_dir=output_dir,
            show_plots=show_plots
        )
        
        # Track results
        file_success = len(vis_files) > 0
        if file_success:
            success_count += 1
            vis_count += len(vis_files)
        else:
            error_count += 1
        
        # Store result
        results[npz_file] = {
            'success': file_success,
            'visualizations': list(vis_files.keys()),
            'visualization_count': len(vis_files)
        }
    
    # Print summary
    print("\nVisualization Summary:")
    print(f"- Total files processed: {len(npz_files)}")
    print(f"- Successful: {success_count}")
    print(f"- Failed: {error_count}")
    print(f"- Total visualizations created: {vis_count}")
    
    return results

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="RNA Feature Visualization Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Create default directories if they don't exist
    default_dir = Path("./data/processed/visualizations")
    default_dir.mkdir(exist_ok=True, parents=True)
    
    # Input options
    parser.add_argument('-f', '--file', help="Path to NPZ file containing RNA features")
    parser.add_argument('-d', '--directory', help="Directory containing NPZ files to process")
    parser.add_argument('-p', '--pattern', default="*_features.npz", 
                      help="File pattern for NPZ files when using --directory")
    
    # Output options
    parser.add_argument('-o', '--output-dir', default="./data/processed/visualizations",
                      help="Directory to save visualization files (default: data/processed/visualizations or same as NPZ files if individual file specified)")
    
    # Visualization options
    parser.add_argument('--show-plots', action='store_true', 
                      help="Show plots interactively (not recommended for batch mode)")
    
    args = parser.parse_args()
    
    # Check dependencies
    if not has_numpy:
        print("ERROR: NumPy is required for this tool")
        sys.exit(1)
    
    if not has_matplotlib:
        print("ERROR: Matplotlib is required for visualization")
        sys.exit(1)
    
    # Determine NPZ files to process
    npz_files = []
    
    if args.file:
        # Single file mode
        if os.path.isfile(args.file):
            npz_files = [args.file]
        else:
            print(f"ERROR: File not found: {args.file}")
            sys.exit(1)
    
    elif args.directory:
        # Directory mode
        if os.path.isdir(args.directory):
            # Find all NPZ files matching the pattern
            pattern = os.path.join(args.directory, args.pattern)
            npz_files = glob.glob(pattern)
            
            if not npz_files:
                print(f"ERROR: No files matching '{args.pattern}' found in {args.directory}")
                sys.exit(1)
        else:
            print(f"ERROR: Directory not found: {args.directory}")
            sys.exit(1)
    
    else:
        # No input provided
        parser.print_help()
        print("\nERROR: You must provide a file or directory to process.")
        sys.exit(1)
    
    # Process the NPZ files
    if len(npz_files) == 1:
        # Single file processing
        visualize_features(
            npz_file=npz_files[0],
            output_dir=args.output_dir,
            show_plots=args.show_plots
        )
    else:
        # Batch processing
        batch_visualize_features(
            npz_files=npz_files,
            output_dir=args.output_dir,
            show_plots=args.show_plots
        )

if __name__ == "__main__":
    main()