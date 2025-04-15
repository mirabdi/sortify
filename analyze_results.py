import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class BenchmarkAnalyzer:
    def __init__(self):
        # Time complexity labels for plotting
        self.complexity_labels = {
            'Quadratic': r'$O(n^2)$',
            'Linearithmic': r'$O(n \log n)$'
        }
        
        # Algorithm groupings
        self.quadratic_algorithms = {
            'Bubble Sort', 'Selection Sort', 'Insertion Sort', 'Cocktail Sort'
        }
        self.linearithmic_algorithms = {
            'Merge Sort', 'Heap Sort', 'Quick Sort', 'Timsort', 'Introsort',
            'Library Sort', 'Tournament Sort', 'Comb Sort'
        }
        
        # Create output directories
        self.time_dir = 'data/time'
        self.memory_dir = 'data/memory'
        os.makedirs(self.time_dir, exist_ok=True)
        os.makedirs(self.memory_dir, exist_ok=True)

    def plot_results(self, df: pd.DataFrame):
        """Generate performance plots from benchmark results."""
        # Plot Time results
        for data_type in df['Data Type'].unique():
            for complexity, algorithms in [
                ('Quadratic', self.quadratic_algorithms),
                ('Linearithmic', self.linearithmic_algorithms)
            ]:
                plt.figure(figsize=(12, 8))
                plot_data = df[(df['Data Type'] == data_type) & (df['Complexity'] == complexity)]
                
                if plot_data.empty:
                    plt.close()
                    continue
                
                for algo in algorithms:
                    algo_data = plot_data[plot_data['Algorithm'] == algo]
                    if not algo_data.empty:
                        times = algo_data['Time (s)'].copy().replace(0, 1e-9)
                        plt.plot(algo_data['Log2_Size'], times, marker='o', 
                                label=f"{algo} ({'Stable' if algo_data['Stable'].iloc[0] else 'Unstable'})")
                
                plt.xlabel('Input Size (log2 n)')
                plt.ylabel('Time (seconds)')
                plt.title(f'{self.complexity_labels[complexity]} Sorting Algorithms - Time\n({data_type} data)')
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Algorithm (Stability)")
                plt.grid(True, which='both', linestyle='--', linewidth=0.5)
                plt.yscale('log')
                plt.xscale('linear')
                
                # Add complexity reference lines
                x_log2 = np.array(sorted(plot_data['Log2_Size'].unique()))
                x_val = 2**x_log2
                min_time = plot_data['Time (s)'].replace(0, np.nan).min()
                if min_time is not np.nan and len(x_val) > 0:
                    if complexity == 'Quadratic':
                        y = (x_val / x_val[0])**2 * min_time
                        plt.plot(x_log2, y, '--', label='O(n²) ref', alpha=0.4, color='gray')
                    else:
                        log_term = np.log2(np.maximum(x_val / x_val[0], 1.0001))
                        y = (x_val / x_val[0]) * log_term * min_time
                        plt.plot(x_log2, y, '--', label='O(n log n) ref', alpha=0.4, color='gray')
                
                min_y_lim = 1e-7
                current_ylim = plt.ylim()
                plt.ylim(min_y_lim, current_ylim[1])
                plt.tight_layout(rect=[0, 0, 0.85, 1])
                
                filename = os.path.join(self.time_dir, f'benchmark_time_{complexity.lower()}_{data_type}.png')
                plt.savefig(filename, bbox_inches='tight', dpi=300)
                print(f"Saved time plot: {filename}")
                plt.close()

        # Plot Memory results
        for data_type in df['Data Type'].unique():
            for complexity, algorithms in [
                ('Quadratic', self.quadratic_algorithms),
                ('Linearithmic', self.linearithmic_algorithms)
            ]:
                plt.figure(figsize=(12, 8))
                plot_data = df[(df['Data Type'] == data_type) & (df['Complexity'] == complexity)]
                
                if plot_data.empty or 'Peak Memory (MiB)' not in plot_data.columns:
                    plt.close()
                    continue

                for algo in algorithms:
                    algo_data = plot_data[plot_data['Algorithm'] == algo]
                    if not algo_data.empty and algo_data['Peak Memory (MiB)'].notnull().any():
                        memory = algo_data['Peak Memory (MiB)'].copy().replace(0, 1e-6)
                        plt.plot(algo_data['Log2_Size'], memory, marker='o',
                                label=f"{algo} ({'Stable' if algo_data['Stable'].iloc[0] else 'Unstable'})")
                
                plt.xlabel('Input Size (log2 n)')
                plt.ylabel('Peak Memory Usage (MiB)')
                plt.title(f'{self.complexity_labels[complexity]} Sorting Algorithms - Memory\n({data_type} data)')
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Algorithm (Stability)")
                plt.grid(True, which='both', linestyle='--', linewidth=0.5)
                plt.xscale('linear')
                plt.tight_layout(rect=[0, 0, 0.85, 1])

                filename = os.path.join(self.memory_dir, f'benchmark_memory_{complexity.lower()}_{data_type}.png')
                plt.savefig(filename, bbox_inches='tight', dpi=300)
                print(f"Saved memory plot: {filename}")
                plt.close()

    def generate_statistics(self, df: pd.DataFrame):
        """Generate detailed statistics for time and memory performance."""
        # Time statistics
        time_stats = []
        for data_type in df['Data Type'].unique():
            for complexity in ['Quadratic', 'Linearithmic']:
                complexity_data = df[(df['Data Type'] == data_type) & (df['Complexity'] == complexity)]
                if complexity_data.empty:
                    continue
                
                for size in sorted(complexity_data['Size'].unique()):
                    size_data = complexity_data[complexity_data['Size'] == size]
                    if size_data.empty:
                        continue
                    
                    # Sort by time for ranking
                    size_data_sorted = size_data.sort_values('Time (s)')
                    
                    for rank, (_, row) in enumerate(size_data_sorted.iterrows(), 1):
                        time_stats.append({
                            'Data Type': data_type,
                            'Complexity': complexity,
                            'Size': size,
                            'Log2_Size': row['Log2_Size'],
                            'Algorithm': row['Algorithm'],
                            'Rank': rank,
                            'Time (s)': row['Time (s)'],
                            'Std Dev Time': row['Std Dev Time'],
                            'Relative Performance': row['Time (s)'] / size_data_sorted['Time (s)'].min()
                        })
        
        # Memory statistics
        memory_stats = []
        for data_type in df['Data Type'].unique():
            for complexity in ['Quadratic', 'Linearithmic']:
                complexity_data = df[(df['Data Type'] == data_type) & (df['Complexity'] == complexity)]
                if complexity_data.empty:
                    continue
                
                for size in sorted(complexity_data['Size'].unique()):
                    size_data = complexity_data[complexity_data['Size'] == size]
                    if size_data.empty:
                        continue
                    
                    # Sort by memory usage for ranking
                    size_data_sorted = size_data.sort_values('Peak Memory (MiB)')
                    
                    for rank, (_, row) in enumerate(size_data_sorted.iterrows(), 1):
                        memory_stats.append({
                            'Data Type': data_type,
                            'Complexity': complexity,
                            'Size': size,
                            'Log2_Size': row['Log2_Size'],
                            'Algorithm': row['Algorithm'],
                            'Rank': rank,
                            'Peak Memory (MiB)': row['Peak Memory (MiB)'],
                            'Std Dev Memory (MiB)': row['Std Dev Memory (MiB)'],
                            'Relative Memory Usage': row['Peak Memory (MiB)'] / size_data_sorted['Peak Memory (MiB)'].min()
                        })
        
        # Save statistics to CSV files
        pd.DataFrame(time_stats).to_csv(os.path.join(self.time_dir, 'time_statistics.csv'), index=False)
        pd.DataFrame(memory_stats).to_csv(os.path.join(self.memory_dir, 'memory_statistics.csv'), index=False)
        
        # Generate summary reports
        self._generate_summary_report(time_stats, 'time')
        self._generate_summary_report(memory_stats, 'memory')

    def _generate_summary_report(self, stats: list, metric_type: str):
        """Generate a summary report for time or memory statistics."""
        df = pd.DataFrame(stats)
        output_dir = self.time_dir if metric_type == 'time' else self.memory_dir
        
        # Group by data type and complexity
        for data_type in df['Data Type'].unique():
            for complexity in ['Quadratic', 'Linearithmic']:
                subset = df[(df['Data Type'] == data_type) & (df['Complexity'] == complexity)]
                if subset.empty:
                    continue
                
                # Calculate average rankings
                avg_rankings = subset.groupby('Algorithm')['Rank'].mean().sort_values()
                
                # Save to file
                filename = os.path.join(output_dir, f'summary_{metric_type}_{complexity.lower()}_{data_type}.txt')
                with open(filename, 'w') as f:
                    f.write(f"Summary Statistics for {metric_type.capitalize()} Performance\n")
                    f.write(f"Data Type: {data_type}\n")
                    f.write(f"Complexity: {complexity}\n\n")
                    
                    f.write("Average Rankings:\n")
                    for algo, rank in avg_rankings.items():
                        f.write(f"{algo}: {rank:.2f}\n")
                    
                    f.write("\nDetailed Statistics:\n")
                    for size in sorted(subset['Size'].unique()):
                        size_data = subset[subset['Size'] == size]
                        f.write(f"\nSize: {size} (log2={size_data['Log2_Size'].iloc[0]})\n")
                        f.write("-" * 50 + "\n")
                        
                        metric_col = 'Time (s)' if metric_type == 'time' else 'Peak Memory (MiB)'
                        std_col = 'Std Dev Time' if metric_type == 'time' else 'Std Dev Memory (MiB)'
                        rel_col = 'Relative Performance' if metric_type == 'time' else 'Relative Memory Usage'
                        
                        for _, row in size_data.sort_values(metric_col).iterrows():
                            f.write(f"{row['Algorithm']}:\n")
                            f.write(f"  {metric_col}: {row[metric_col]:.6f} (±{row[std_col]:.6f})\n")
                            f.write(f"  {rel_col}: {row[rel_col]:.2f}x\n")
                            f.write(f"  Rank: {row['Rank']}\n")

def main():
    # Check if results file exists
    results_file = 'data/benchmark_results.csv'
    if not os.path.exists(results_file):
        print(f"Error: Results file not found at {results_file}")
        print("Please run run_benchmarks.py first to generate benchmark results.")
        return

    # Read results
    print("Reading benchmark results...")
    df = pd.read_csv(results_file)
    
    # Create analyzer and generate visualizations
    analyzer = BenchmarkAnalyzer()
    
    print("\nGenerating plots and statistics...")
    analyzer.plot_results(df)
    analyzer.generate_statistics(df)
    
    print("\nAnalysis complete! Results saved to:")
    print("- data/time/ (time performance analysis)")
    print("- data/memory/ (memory usage analysis)")

if __name__ == "__main__":
    main() 