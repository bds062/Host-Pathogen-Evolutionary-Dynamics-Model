# models_plotting.py
import math
import time
import os
import hashlib
import json
from datetime import datetime
import numpy as np
from multiprocessing import Pool

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from timing import (
    cores, floweringS, vegetatingS, floweringI, vegetatingI,
    germination, tempVector, temp
)
from model import Model

def _compute_ess(model_kwargs, flowering_values, num_processes):
    """Run one raster (infected_time vs infected_time2) and return the ESS flowering time.

    Builds the full argument list, lets run_single_simulation skip the lower triangle
    (infected_time < infected_time2) as usual, then fills the symmetric values and
    returns the column with the minimum column sum.
    """
    n = len(flowering_values)
    heatmap = np.zeros((n, n))

    args_list = [
        (p1, p2, 'infected_time', 'infected_time2', model_kwargs, (i, j))
        for i, p1 in enumerate(flowering_values)
        for j, p2 in enumerate(flowering_values)
    ]

    with Pool(processes=num_processes) as pool:
        for (i, j), result in pool.imap(run_single_simulation, args_list):
            if result is not None:
                heatmap[i, j] = result

    # Fill the upper triangle (i < j, which was skipped) via symmetry
    for i in range(n):
        for j in range(i + 1, n):
            heatmap[i, j] = 1.0 - heatmap[j, i]

    ess_idx = int(np.argmin(np.sum(heatmap, axis=0)))
    return float(flowering_values[ess_idx])


def run_single_simulation(args):
    param1_val, param2_val, param1_name, param2_name, model_kwargs, index_info = args
    model = Model(**model_kwargs)

    if param1_val is not None and param1_name is not None:
        setattr(model, param1_name, param1_val)
    if param2_val is not None and param2_name is not None:
        setattr(model, param2_name, param2_val)

    if (param1_name == 'infected_time' and param2_name == 'infected_time2') or \
       (param1_name == 'infected_time2' and param2_name == 'infected_time'):
        if param1_val < param2_val:
            # print(f"Skipping {param1_name}={param1_val}, {param2_name}={param2_val} (will use symmetry)")
            return (index_info, None)

    solution = model.run_sim()
    model.eval = 6
    result = model.evaluate(solution)

    # print(f"Completed simulation for {param1_name}={param1_val}, {param2_name}={param2_val} with result={result}")
    return (index_info, result)


def compute_curve_peaks(x_values, curves, labels=None):
    """
    Find the peak (max-y) location and height of each curve in `curves`.

    Args:
        x_values: 1D array shared by every curve (e.g. germination_time sweep values).
        curves:   List of 1D y-arrays, one per curve.
        labels:   Optional list of names, one per curve.

    Returns:
        dict with per-curve 'peak_days' (x at the max) and 'peak_values' (the max y),
        plus 'mean_peak_day'/'std_peak_day'/'var_peak_day' and
        'mean_peak_value'/'std_peak_value'/'var_peak_value' across all curves.
    """
    x_values = np.asarray(x_values, dtype=float)
    peak_days, peak_values = [], []
    for y in curves:
        y = np.asarray(y, dtype=float)
        if np.all(np.isnan(y)):
            peak_days.append(np.nan)
            peak_values.append(np.nan)
            continue
        idx = int(np.nanargmax(y))
        peak_days.append(float(x_values[idx]))
        peak_values.append(float(y[idx]))

    peak_days = np.array(peak_days)
    peak_values = np.array(peak_values)
    valid = ~np.isnan(peak_days)

    return {
        'labels': list(labels) if labels is not None else [f"curve_{i}" for i in range(len(curves))],
        'peak_days': peak_days,
        'peak_values': peak_values,
        'mean_peak_day': float(np.mean(peak_days[valid])) if valid.any() else float('nan'),
        'std_peak_day': float(np.std(peak_days[valid])) if valid.any() else float('nan'),
        'var_peak_day': float(np.var(peak_days[valid])) if valid.any() else float('nan'),
        'mean_peak_value': float(np.mean(peak_values[valid])) if valid.any() else float('nan'),
        'std_peak_value': float(np.std(peak_values[valid])) if valid.any() else float('nan'),
        'var_peak_value': float(np.var(peak_values[valid])) if valid.any() else float('nan'),
    }


def _peak_summary_text(fig_name, peak_info, x_label='day of year'):
    """Build the multi-line peak-summary report shared by print_peak_summary() and
    save_peak_summary_txt()."""
    lines = [f"{fig_name} — peak summary ({x_label}):"]
    for label, day, val in zip(peak_info['labels'], peak_info['peak_days'], peak_info['peak_values']):
        lines.append(f"  {label:45s} peak {x_label} = {day:7.1f}   peak value = {val:.4f}")
    lines.append(f"  {'-' * 70}")
    lines.append(f"  Mean peak {x_label} = {peak_info['mean_peak_day']:.2f}   "
                 f"Std = {peak_info['std_peak_day']:.2f}   Var = {peak_info['var_peak_day']:.2f}")
    lines.append(f"  Mean peak value{'':<{max(len(x_label) - 5, 0)}} = {peak_info['mean_peak_value']:.4f}   "
                 f"Std = {peak_info['std_peak_value']:.4f}   Var = {peak_info['var_peak_value']:.4f}")
    return "\n".join(lines)


def print_peak_summary(fig_name, peak_info, x_label='day of year'):
    """Pretty-print the per-curve and aggregate peak stats from compute_curve_peaks()."""
    print("\n" + _peak_summary_text(fig_name, peak_info, x_label))


def save_peak_summary_txt(path, fig_name, peak_info, x_label='day of year'):
    """
    Write (or update) this figure's peak-summary block inside a shared summary.txt file.

    The file holds one "=== <fig_name> ===" block per figure; calling this again with
    the same fig_name replaces just that block, leaving other figures' blocks intact —
    so figures can be (re)run independently and the summary stays complete and current.
    """
    path = os.fspath(path)
    header = f"=== {fig_name} ==="
    block = (f"{header}\n"
             f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
             f"{_peak_summary_text(fig_name, peak_info, x_label)}")

    blocks = {}
    order = []
    if os.path.exists(path):
        with open(path) as f:
            content = f.read()
        for raw_block in content.split("\n\n"):
            raw_block = raw_block.strip("\n")
            if not raw_block.strip():
                continue
            block_header = raw_block.splitlines()[0].strip()
            blocks[block_header] = raw_block
            order.append(block_header)

    if header not in blocks:
        order.append(header)
    blocks[header] = block

    known_order = [f"=== Fig {n} ===" for n in ('1', '2', '3', '4')]
    ordered_headers = [h for h in known_order if h in blocks] + \
                       [h for h in order if h not in known_order]

    with open(path, 'w') as f:
        f.write("\n\n".join(blocks[h] for h in ordered_headers) + "\n")
    print(f"Peak summary written to: {path}")


def save_figure_embedding(path, fig_name, x_name, x_values, curves, labels, extra_metadata=None):
    """
    Save a figure's underlying sweep curves ("embedding") plus their peak statistics
    to a .npz file, so the data can be reloaded later (via load_figure_embedding) without
    rerunning the simulations. Written atomically.

    Returns the peak_info dict (see compute_curve_peaks) for convenience.
    """
    peak_info = compute_curve_peaks(x_values, curves, labels)

    metadata = {
        'fig_name': fig_name,
        'x_name': x_name,
        'labels': list(labels),
        'mean_peak_day': peak_info['mean_peak_day'],
        'std_peak_day': peak_info['std_peak_day'],
        'var_peak_day': peak_info['var_peak_day'],
        'mean_peak_value': peak_info['mean_peak_value'],
        'std_peak_value': peak_info['std_peak_value'],
        'var_peak_value': peak_info['var_peak_value'],
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    path = os.fspath(path)
    tmp_path = path + '.tmp.npz'
    np.savez(
        tmp_path,
        x_values=np.asarray(x_values, dtype=float),
        curves=np.asarray(curves, dtype=float),
        labels=np.array(list(labels)),
        peak_days=peak_info['peak_days'],
        peak_values=peak_info['peak_values'],
        metadata=np.array(json.dumps(metadata)),
    )
    os.replace(tmp_path, path)
    print(f"Embedding saved to: {path}")
    return peak_info


def load_figure_embedding(path):
    """Reload a .npz embedding written by save_figure_embedding()."""
    with np.load(path, allow_pickle=False) as data:
        out = {
            'x_values': np.asarray(data['x_values']),
            'curves': np.asarray(data['curves']),
            'labels': [str(l) for l in data['labels']],
            'peak_days': np.asarray(data['peak_days']),
            'peak_values': np.asarray(data['peak_values']),
        }
        if 'metadata' in data.files:
            out['metadata'] = json.loads(str(np.asarray(data['metadata']).item()))
    return out


def _load_cached_curves(path, x_values, expected_labels):
    """
    Try to reuse a previously saved embedding instead of rerunning sweeps.

    Returns the list of curves (in expected_labels order) if the cached embedding's
    x-axis and labels match exactly, else None (caller should rerun the sweep).
    """
    if path is None or not os.path.exists(path):
        return None
    try:
        emb = load_figure_embedding(path)
    except (FileNotFoundError, KeyError, ValueError, OSError, json.JSONDecodeError):
        return None

    if emb['labels'] != list(expected_labels):
        return None
    x_values = np.asarray(x_values, dtype=float)
    if len(emb['x_values']) != len(x_values) or not np.allclose(emb['x_values'], x_values):
        return None

    return [row for row in emb['curves']]


def summarize_peak_embeddings(paths, source_labels=None, show_plot=True):
    """
    Load multiple saved figure embeddings (fig1/fig2/fig4 results_save_path outputs)
    and display combined peak-day statistics, per source figure and overall.

    Args:
        paths:         List of .npz embedding paths.
        source_labels: Optional display name per path (defaults to the file's stem).
        show_plot:      If True, draws a strip plot of peak days grouped by source figure.

    Returns:
        dict with 'source_labels', 'peak_days_by_source', and 'combined_peak_days'.
    """
    paths = list(paths)
    if source_labels is None:
        source_labels = [os.path.splitext(os.path.basename(p))[0] for p in paths]

    all_days = []
    print("\nCombined peak summary across figures:")
    for path, name in zip(paths, source_labels):
        emb = load_figure_embedding(path)
        days = emb['peak_days']
        valid = days[~np.isnan(days)]
        all_days.append(valid)
        mean = np.mean(valid) if valid.size else float('nan')
        std = np.std(valid) if valid.size else float('nan')
        print(f"  {name:30s} n={len(valid):2d}  mean={mean:7.2f}  std={std:7.2f}")

    combined = np.concatenate(all_days) if all_days else np.array([])
    if combined.size:
        print(f"  {'-' * 60}")
        print(f"  {'ALL FIGURES':30s} n={len(combined):2d}  "
              f"mean={np.mean(combined):7.2f}  std={np.std(combined):7.2f}")

    if show_plot and combined.size:
        plt.figure(figsize=(8, 4))
        for i, (name, days) in enumerate(zip(source_labels, all_days)):
            if len(days) == 0:
                continue
            jitter = (np.random.rand(len(days)) - 0.5) * 0.15
            plt.scatter(np.full(len(days), i) + jitter, days, alpha=0.8, s=40, zorder=3)
            plt.errorbar(i, np.mean(days), yerr=np.std(days), fmt='D', color='black',
                         markersize=6, capsize=4, zorder=5)
        plt.xticks(range(len(source_labels)), source_labels, rotation=20, ha='right')
        plt.ylabel('Peak day of year')
        plt.title('Peak timing across figures')
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

    return {'source_labels': source_labels, 'peak_days_by_source': all_days, 'combined_peak_days': combined}


class GraphModel:
    def __init__(self, model):
        self.model = model
        self.colSj = '#00ff4c'
        self.colSv = '#1fad4a'
        self.colSf = '#1f6133'
        self.colIj = '#ff000d'
        self.colIv = '#b81820'
        self.colIf = '#7a2328'
        self.colTot = "#000000"
        self.colSd = "#000000"
        self.colIj2 = "#68cffb"
        self.colIv2 = "#3E66BC"
        self.colIf2 = "#0000ec"

    def graph(self):
        solution = self.model.run_sim()
        print(f"Success value is {self.model.evaluate(solution)}. Final values are:")

        final_values = solution.y[:, -1]
        class_names = [
            "Susceptible Juveniles", "Susceptible Vegetatives", "Susceptible Flowerings",
            "Infected Juveniles", "Infected Vegetatives", "Infected Flowerings",
            "Seeds", "Infected2 Juveniles", "Infected2 Vegetatives", "Infected2 Flowerings"
        ]
        for name, value in zip(class_names, final_values):
            print(f"{name}: {value:.4f}")

        time_points = self.model.solution.t
        Sj, Sv, Sf, Ij, Iv, If, Sd, Ij2, Iv2, If2 = self.model.solution.y
        total_I1 = Ij + Iv + If
        total_I2 = Ij2 + Iv2 + If2
        total_I = total_I1 + total_I2
        total_S = Sj + Sv + Sf
        I1_percent = (total_I1 / total_I) * 100
        S_percent = (total_S / (total_S + total_I)) * 100

        fig, populationGraph = plt.subplots(figsize=(10, 10))
        populationGraph.set_title('Population Levels (SI)')
        # populationGraph.plot(time_points, Sv, label='SV', color=self.colSv)
        # populationGraph.plot(time_points, Sf, label='Sf', color=self.colSf)
        populationGraph.plot(time_points, total_S, label='Susceptibles', color=self.colSv)
        populationGraph.plot(time_points, total_I1, label='Infected1', color=self.colIv)
        populationGraph.plot(time_points, total_I2, label='Infected2', color=self.colIv2)
        populationGraph.set_xlabel('Time (years)')
        populationGraph.set_ylabel('Abundance')
        # populationGraph.set_ylim(0, 35)
        populationGraph.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 365:.1f}"))
        populationGraph.legend()

        fig, custom2 = plt.subplots(figsize=(10, 10))
        custom2.set_title('Custom Growth Rate Functions')

        year_time = np.linspace(0, 365, 1000)

        susceptible_offset = 182.5 - self.model.susceptible_time
        infected_offset = 182.5 - self.model.infected_time
        infected_offset2 = 182.5 - self.model.infected_time2
        germination_offset = 182.5 - self.model.germination_time

        custom2.plot(year_time, germination(tempVector(year_time + germination_offset)),
                     label=f'Germination (peak at day {self.model.germination_time:.1f})')
        custom2.plot(year_time, floweringS(tempVector(year_time + susceptible_offset)),
                     label=f'Susceptible Flowering (peak at day {self.model.susceptible_time:.1f})',
                     color=self.colSf)
        custom2.plot(year_time, floweringI(tempVector(year_time + infected_offset)),
                     label=f'Infected Flowering (peak at day {self.model.infected_time:.1f})',
                     color=self.colIf)
        custom2.plot(year_time, floweringI(tempVector(year_time + infected_offset2)),
                     label=f'Infected2 Flowering (peak at day {self.model.infected_time2:.1f})',
                     color=self.colIf2)
        custom2.set_xlabel('Day of Year')
        custom2.set_ylabel('Transition Rate')
        custom2.set_xlim(0, 365)
        custom2.legend()
        custom2.grid(True, alpha=0.3)

        fig, prevalenceGraph = plt.subplots(figsize=(10, 10))
        prevalenceGraph.set_title('Genotype Frequency Over Time')
        #Allelic invasion
        # prevalenceGraph.fill_between(time_points, 0, I1_percent, color=self.colIf, alpha=0.7, label='Infected 1')
        # prevalenceGraph.fill_between(time_points, I1_percent, 100, color=self.colIf2, alpha=0.7, label='Infected 2')
        #Normal
        prevalenceGraph.fill_between(time_points, 0, S_percent, color=self.colSf, alpha=0.7, label='Susceptible')
        prevalenceGraph.fill_between(time_points, S_percent, 100, color=self.colIf, alpha=0.7, label='Infecteds')
        prevalenceGraph.set_xlabel('time')
        prevalenceGraph.set_ylabel('percent prevalence')
        prevalenceGraph.set_ylim(0, 100)
        prevalenceGraph_right = prevalenceGraph.twinx()
        prevalenceGraph_right.set_ylabel('percent prevalence')
        prevalenceGraph_right.set_ylim(100, 0)
        prevalenceGraph.legend()
        prevalenceGraph.grid(True, alpha=0.3)

class RasterModel:
    def __init__(self, model, param1_name, param2_name, param1_values, param2_values):
        self.model = model
        self.param1_name = param1_name
        self.param2_name = param2_name
        self.param1_values = param1_values
        self.param2_values = param2_values
        
        # Initialize data structures
        # Always create plot_data_1d for 1D sweeps
        if param1_values is not None:
            self.plot_data_1d = np.zeros(max(len(param1_values), len(param2_values) if param2_values is not None else 0))
        
        # Create heatmap_data if both parameters are provided
        if param1_values is not None and param2_values is not None:
            self.heatmap_data = np.zeros((len(param1_values), len(param2_values)))

    def sweep_1d(self, param_num=1, num_processes=None):
        """
        Perform a 1D parameter sweep, varying either param1 or param2.
        
        Args:
            param_num: Which parameter to vary (1 or 2). Default is 1.
            num_processes: Number of processes to use. If None, uses all available cores.
        
        Returns:
            Array of evaluation results for each parameter value
        """
        if param_num not in [1, 2]:
            raise ValueError("param_num must be 1 or 2")
        
        # Determine which parameter to vary
        if param_num == 1:
            if self.param1_name is None or self.param1_values is None:
                raise ValueError("param1_name and param1_values must be set to sweep parameter 1")
            self.param_name = self.param1_name
            self.param_values = self.param1_values
            fixed_param_name = self.param2_name
        else:  # param_num == 2
            if self.param2_name is None or self.param2_values is None:
                raise ValueError("param2_name and param2_values must be set to sweep parameter 2")
            self.param_name = self.param2_name
            self.param_values = self.param2_values
            fixed_param_name = self.param1_name
        
        if num_processes is None:
            num_processes = cores
        
        param_values = self.param_values
        param_name = self.param_name

        total_iterations = len(param_values)
        print(f"Starting 1D parameter sweep with {num_processes} processes...")
        print(f"Sweeping {param_name}, total iterations: {total_iterations}")
        
        # Get model kwargs to pass to workers
        model_kwargs = {
            'Bj': self.model.Bj,
            'Bv': self.model.Bv,
            'Bf': self.model.Bf,
            'births': self.model.births,
            'gamma': self.model.gamma,
            'death': self.model.death,
            'maturity': self.model.maturity,
            'g': self.model.g,
            'susceptible_time': self.model.susceptible_time,
            'infected_time': self.model.infected_time,
            'infected_time2': self.model.infected_time2,
            'germination_time': self.model.germination_time,
            'eval': self.model.eval,
            'X_0': self.model.X_0,
            't': self.model.t
        }
        
        # Prepare arguments for all simulations
        args_list = []
        if param_num == 1:
            # Get the fixed value for param2 from the model
            fixed_param2_val = getattr(self.model, self.param2_name) if self.param2_name else None
            for i, param_val in enumerate(param_values):
                # Include index i so we know where to place the result
                args_list.append((param_val, fixed_param2_val, self.param1_name, 
                                self.param2_name, model_kwargs, i))
        else:  # param_num == 2
            # Get the fixed value for param1 from the model
            fixed_param1_val = getattr(self.model, self.param1_name) if self.param1_name else None
            for i, param_val in enumerate(param_values):
                # Include index i so we know where to place the result
                args_list.append((fixed_param1_val, param_val, self.param1_name, 
                                self.param2_name, model_kwargs, i))
        
        # Run simulations in parallel with progress tracking
        max_eval = -math.inf
        max_param = 0
        
        import time
        start_time = time.time()
        
        # Store which parameter we're sweeping for later use in plot_1d
        self.last_sweep_param_num = param_num
        self.last_sweep_param_name = param_name
        self.last_sweep_param_values = param_values
        
        with Pool(processes=num_processes) as pool:
            completed = 0
            last_update = 0
            update_interval = 25  # Update every 25%
            
            print(f"Progress: 0% complete")
            
            # Use imap for ordered results (could also use imap_unordered with index tracking)
            for index_info, result in pool.imap(run_single_simulation, args_list):
                i = index_info  # For 1D sweep, index_info is just i
                
                # Place result in correct position
                self.plot_data_1d[i] = result
                
                if result > max_eval:
                    max_eval = result
                    max_param = param_values[i]
                
                completed += 1
                
                # Calculate progress percentage
                progress = (completed / total_iterations) * 100
                
                # Print update every interval
                if progress >= last_update + update_interval or completed == total_iterations:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    remaining = (total_iterations - completed) / rate if rate > 0 else 0
                    
                    print(f"Progress: {int(progress)}% complete "
                          f"({completed}/{total_iterations}) - "
                          f"Elapsed: {elapsed:.1f}s - "
                          f"Est. remaining: {remaining:.1f}s")
                    
                    last_update = int(progress / update_interval) * update_interval
        
        # Trim plot_data_1d to actual size used
        self.plot_data_1d = self.plot_data_1d[:len(param_values)]
        
        total_time = time.time() - start_time
        print(f"\n1D sweep complete in {total_time:.1f}s!")
        # print(f"Max value: {max_eval}")
        # print(f"At {param_name} = {max_param}")
        
        return self.plot_data_1d

    def plot_1d(self, figsize=(10, 6), color='#2563eb', linewidth=2.5):
        """
        Plot the 1D parameter sweep results as a line plot.
        
        Args:
            figsize: Figure size tuple (width, height)
            color: Line color
            linewidth: Line width
        """
        plt.figure(figsize=figsize)
        plt.plot(self.last_sweep_param_values, self.plot_data_1d, color=color, linewidth=linewidth)
        plt.ylim(0, 0.5) #just for prevalence phase 2
        plt.xlabel(self.last_sweep_param_name.replace('_', ' ').title(), fontsize=12)
        plt.ylabel('Equilibrium Prevalence', fontsize=12)
        plt.title(f'Equilibrium Prevalence vs {self.last_sweep_param_name.replace("_", " ").title()}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print(f"\nSummary Statistics:")
        print(f"Min: {np.min(self.plot_data_1d):.4f} at {self.last_sweep_param_name} = {self.last_sweep_param_values[np.argmin(self.plot_data_1d)]:.2f}")
        print(f"Max: {np.max(self.plot_data_1d):.4f} at {self.last_sweep_param_name} = {self.last_sweep_param_values[np.argmax(self.plot_data_1d)]:.2f}")
        print(f"Mean: {np.mean(self.plot_data_1d):.4f}")
        print(f"Std: {np.std(self.plot_data_1d):.4f}")

    def raster(self, num_processes=None):
        """
        Parallel version of raster using multiprocessing.
        
        Args:
            num_processes: Number of processes to use. If None, uses all available cores.
        """
        if self.param1_values is None or self.param2_values is None:
            raise ValueError("Both param1_values and param2_values must be set for 2D raster. Use sweep_1d() for 1D sweeps.")
        
        if num_processes is None:
            num_processes = cores
        
        total_iterations = len(self.param1_values) * len(self.param2_values)
        print(f"Starting parallel raster process with {num_processes} processes...")
        print(f"Total iterations: {total_iterations}")
        
        # Get model kwargs to pass to workers
        model_kwargs = {
            'Bj': self.model.Bj,
            'Bv': self.model.Bv,
            'Bf': self.model.Bf,
            'births': self.model.births,
            'gamma': self.model.gamma,
            'death': self.model.death,
            'maturity': self.model.maturity,
            'g': self.model.g,
            'susceptible_time': self.model.susceptible_time,
            'infected_time': self.model.infected_time,
            'infected_time2': self.model.infected_time2,
            'germination_time': self.model.germination_time,
            'eval': self.model.eval,
            'X_0': self.model.X_0,
            't': self.model.t
        }
        
        # Prepare arguments for all simulations
        args_list = []
        for i, param1_val in enumerate(self.param1_values):
            for j, param2_val in enumerate(self.param2_values):
                # Include (i, j) tuple so we know where to place the result
                args_list.append((param1_val, param2_val, self.param1_name, 
                                self.param2_name, model_kwargs, (i, j)))
        
        # Run simulations in parallel with progress tracking
        max_eval = -math.inf
        max_p1 = 0
        max_p2 = 0
        
        import time
        start_time = time.time()
        
        with Pool(processes=num_processes) as pool:
            completed = 0
            last_update = 0
            update_interval = 25  # Update every 25%
            
            print(f"Progress: 0% complete")
            
            # Use imap for ordered results
            for index_info, result in pool.imap(run_single_simulation, args_list):
                i, j = index_info  # Unpack the indices
                
                # Place result in correct position
                self.heatmap_data[i, j] = result
                
                if result is not None and result > max_eval:
                    max_eval = result
                    max_p1 = self.param1_values[i]
                    max_p2 = self.param2_values[j]
                
                completed += 1
                
                # Calculate progress percentage
                progress = (completed / total_iterations) * 100
                
                # Print update every interval
                if progress >= last_update + update_interval or completed == total_iterations:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    remaining = (total_iterations - completed) / rate if rate > 0 else 0
                    
                    print(f"Progress: {int(progress)}% complete "
                          f"({completed}/{total_iterations}) - "
                          f"Elapsed: {elapsed:.1f}s - "
                          f"Est. remaining: {remaining:.1f}s")
                    
                    last_update = int(progress / update_interval) * update_interval
        
        # Fill in symmetric values for competing genotype plots
        if (self.param1_name == 'infected_time' and self.param2_name == 'infected_time2') or \
           (self.param1_name == 'infected_time2' and self.param2_name == 'infected_time'):
            print("\nFilling symmetric values or running missing simulations...")
            
            for i, param1_val in enumerate(self.param1_values):
                for j, param2_val in enumerate(self.param2_values):
                    
                    cur_val = self.heatmap_data[i, j]
                    
                    # 1. Check if the current point is empty/skipped
                    if cur_val is None or np.isnan(cur_val) or cur_val == 0.0:
                        
                        # 2. Safely try to get the mirrored value (avoids IndexError on non-square arrays)
                        mirrored_value = None
                        if j < len(self.param1_values) and i < len(self.param2_values):
                            mirrored_value = self.heatmap_data[j, i]
                            
                        # 3. If the mirror has data, use the reflection
                        if mirrored_value is not None and not np.isnan(mirrored_value) and mirrored_value != 0.0:
                            # print(f"Mirroring point for ({param1_val}, {param2_val})")
                            self.heatmap_data[i, j] = 1.0 - mirrored_value
                            
                        # 4. FALLBACK: No mirror exists. Run the simulation directly here.
                        else:
                            print(f"No mirror for ({param1_val:.1f}, {param2_val:.1f}). Running simulation...")
                            
                            # Create a localized model to bypass the auto-skip in the worker function
                            fallback_model = Model(**model_kwargs)
                            setattr(fallback_model, self.param1_name, param1_val)
                            setattr(fallback_model, self.param2_name, param2_val)
                            
                            solution = fallback_model.run_sim()
                            fallback_model.eval = 6
                            sim_result = fallback_model.evaluate(solution)
                            
                            self.heatmap_data[i, j] = sim_result
            
            # Find ESS (Evolutionarily Stable Strategy) - column with lowest sum
            column_sums = np.sum(self.heatmap_data, axis=0)
            ess_column_idx = np.argmin(column_sums)
            ess_flowering_time = self.param2_values[ess_column_idx]
            print(f"\nESS (Evolutionarily Stable Strategy) found:")
            print(f"  Column index: {ess_column_idx}")
            print(f"  Flowering time: {ess_flowering_time:.2f} days")
            print(f"  Column sum: {column_sums[ess_column_idx]:.4f}")
            
            # Store ESS info for plotting
            self.ess_column_idx = ess_column_idx
            self.ess_flowering_time = ess_flowering_time
            self.is_genotype_competition = True
        else:
            self.is_genotype_competition = False
        
        total_time = time.time() - start_time
        print(f"\nRaster process complete in {total_time:.1f}s!")
        # print(f"Max value: {max_eval}")
        # print(f"At {self.param1_name} = {max_p1} and {self.param2_name} = {max_p2}")
        
        return self.heatmap_data

    def plot_heatmap(self):
        if self.param1_values is None or self.param2_values is None:
            raise ValueError("Both parameters must be set for heatmap. Use plot_1d() for 1D sweeps.")
        
        # Check if this is a genotype competition plot
        is_genotype = hasattr(self, 'is_genotype_competition') and self.is_genotype_competition
        
        plt.figure(figsize=(8, 6))
        
        if is_genotype:
            # Use diverging colormap for genotype competition
            # Red = Genotype 1 wins (high prevalence), Blue = Genotype 2 wins (low prevalence)
            plt.imshow(
                self.heatmap_data,
                extent=[min(self.param2_values), max(self.param2_values),
                        min(self.param1_values), max(self.param1_values)],
                origin="lower",
                aspect="auto",
                cmap="RdBu_r",  # _r reverses it so Red=high, Blue=low
                vmin=0.0,
                vmax=1.0,
            )
            cbar = plt.colorbar(label="Genotype Abundance")
            
            # Determine which genotype is which based on parameter names
            if self.param1_name == 'infected_time':
                genotype1_axis = 'y'
                genotype2_axis = 'x'
            else:  # param1_name == 'infected_time2'
                genotype1_axis = 'x'
                genotype2_axis = 'y'
            
            # Set axis labels with colors matching the colormap
            # Red for Genotype 1, Blue for Genotype 2
            if genotype1_axis == 'y':
                plt.xlabel("Genotype 2 Flowering Time (days)", fontsize=12, color='#2166AC')  # Blue
                plt.ylabel("Genotype 1 Flowering Time (days)", fontsize=12, color='#B2182B')  # Red
            else:
                plt.xlabel("Genotype 1 Flowering Time (days)", fontsize=12, color='#B2182B')  # Red
                plt.ylabel("Genotype 2 Flowering Time (days)", fontsize=12, color='#2166AC')  # Blue
            
            plt.title("Genotype Competition: Prevalence Across Flowering Times", fontsize=14)
            
        else:
            # Standard viridis colormap for non-competition plots
            plt.imshow(
                self.heatmap_data,
                extent=[min(self.param2_values), max(self.param2_values),
                        min(self.param1_values), max(self.param1_values)],
                origin="lower",
                aspect="auto",
                cmap="viridis",
            )
            plt.colorbar(label="Pathogen Prevalence")
            plt.xlabel(f"{self.param2_name.replace('_', ' ').title()}")
            plt.ylabel(f"{self.param1_name.replace('_', ' ').title()}")
            plt.title("Pathogen Prevalence Across Varied Seasonal Peaks")
        
        plt.show()

    def plot_fig1(self, germination_values, maturity_values, bj_values,
              maturity_labels=None, bj_labels=None,
              colors=None, figsize=(10, 5), num_processes=None,
              ylim=None, save_path=None, dpi=300, results_save_path=None,
              use_saved_embedding=False):
        """
        Publication-ready two-panel figure. Each panel sweeps germination_time on the
        x-axis with three maturation rates overlaid as lines. The two panels correspond
        to two different transmission rates (Bj). A dotted vertical line marks peak
        flowering (day 182.5), and a gray background shading reflects the flowering
        intensity (floweringS) across the germination-time axis.

        After the sweeps complete, the peak (max-prevalence) germination day of each of
        the 6 curves is printed along with its mean/stdev across curves (see
        compute_curve_peaks / print_peak_summary). If results_save_path is given, the
        curves and peak stats are also saved as a reloadable embedding
        (see save_figure_embedding / load_figure_embedding / summarize_peak_embeddings).

        Args:
            germination_values: Array of germination_time values to sweep (x-axis).
            maturity_values:    List of exactly 3 maturity rates (per-day) for the line overlay.
            bj_values:          List of exactly 2 Bj values, one per panel.
            maturity_labels:    Optional list of 3 legend labels for maturity lines.
            bj_labels:          Optional list of 2 panel subtitle strings (e.g. ["β_J = 0.1", ...]).
            colors:             Optional list of 3 line colors for the maturity lines.
            figsize:            Figure size tuple. Default (10, 5) suits 2 wide panels.
            ylim:               Optional (ymin, ymax) tuple applied to all panels.
            save_path:          If provided, saves figure to this path (e.g. 'fig1.pdf').
            dpi:                Resolution for saved figure. Default 300.
            num_processes:      Parallel processes. Defaults to cores.
            results_save_path:  If provided, saves the sweep curves + peak stats to this
                                 .npz path (e.g. 'fig1_embedding.npz').
            use_saved_embedding: If True and results_save_path exists with matching
                                 germination_values/labels, skip the sweep entirely and
                                 rebuild the figure from the cached embedding.
        """
        if len(maturity_values) != 3:
            raise ValueError("maturity_values must contain exactly 3 values")
        if len(bj_values) != 2:
            raise ValueError("bj_values must contain exactly 2 values")

        if colors is None:
            colors = ['#2563eb', '#dc2626', '#16a34a']
        if maturity_labels is None:
            maturity_labels = [f"Maturity = {v * 365:.1f} yr⁻¹" for v in maturity_values]
        if bj_labels is None:
            bj_labels = [f"$\\beta_J$ = {bj * 365:.2f} yr⁻¹" for bj in bj_values]
        if num_processes is None:
            num_processes = cores

        germination_values = np.asarray(germination_values)
        panel_labels = ['A', 'B']

        flat_labels = [f"{bj_label} | {mat_label}"
                        for bj_label in bj_labels
                        for mat_label in maturity_labels]

        # ── Run all sweeps (or reuse a cached embedding) ────────────────────────────
        # Shape: [n_panels, n_maturity_values, n_germination_values]
        all_results = None
        reused = False

        if use_saved_embedding:
            cached_curves = _load_cached_curves(results_save_path, germination_values, flat_labels)
            if cached_curves is not None:
                n_mat = len(maturity_values)
                all_results = [cached_curves[i * n_mat:(i + 1) * n_mat] for i in range(len(bj_values))]
                reused = True
                print(f"Reusing cached embedding from: {results_save_path}")
            else:
                print("No matching cached embedding found; running sweeps.")

        if all_results is None:
            all_results = []

            for panel_idx, bj_val in enumerate(bj_values):
                panel_results = []

                for maturity_val in maturity_values:
                    print(f"\n--- Panel {panel_labels[panel_idx]}: "
                        f"Bj={bj_val * 365:.3f}/yr, maturity={maturity_val * 365:.2f}/yr ---")

                    model_kwargs = {
                        'Bj': bj_val,
                        'Bv': self.model.Bv,
                        'Bf': self.model.Bf,
                        'births': self.model.births,
                        'gamma': self.model.gamma,
                        'death': self.model.death,
                        'maturity': maturity_val,
                        'g': self.model.g,
                        'susceptible_time': self.model.susceptible_time,
                        'infected_time': self.model.infected_time,
                        'infected_time2': self.model.infected_time2,
                        'germination_time': self.model.germination_time,
                        'eval': self.model.eval,
                        'X_0': self.model.X_0,
                        't': self.model.t
                    }

                    args_list = [
                        (germ_val, None, 'germination_time', None, model_kwargs, i)
                        for i, germ_val in enumerate(germination_values)
                    ]

                    sweep_results = np.zeros(len(germination_values))
                    total = len(germination_values)
                    completed = 0
                    start_time = time.time()
                    last_update = 0

                    with Pool(processes=num_processes) as pool:
                        for index_info, result in pool.imap(run_single_simulation, args_list):
                            sweep_results[index_info] = result if result is not None else 0.0
                            completed += 1
                            progress = (completed / total) * 100
                            if progress >= last_update + 25 or completed == total:
                                elapsed = time.time() - start_time
                                rate = completed / elapsed if elapsed > 0 else 1
                                remaining = (total - completed) / rate
                                print(f"  {int(progress)}% ({completed}/{total}) — "
                                    f"{elapsed:.1f}s elapsed, ~{remaining:.1f}s remaining")
                                last_update = int(progress / 25) * 25

                    panel_results.append(sweep_results)

                all_results.append(panel_results)

        # ── Peak summary across all 6 curves ────────────────────────────────────────
        flat_curves = [curve for panel in all_results for curve in panel]
        peak_info = compute_curve_peaks(germination_values, flat_curves, flat_labels)
        print_peak_summary("Fig 1", peak_info, x_label='germination day')

        if results_save_path is not None and not reused:
            save_figure_embedding(
                results_save_path, fig_name='fig1', x_name='germination_time',
                x_values=germination_values, curves=flat_curves, labels=flat_labels,
                extra_metadata={'bj_values': list(map(float, bj_values)),
                                 'maturity_values': list(map(float, maturity_values))},
            )

        # ── Precompute flowering intensity across germination-time axis ────────────
        # floweringS takes a temperature, and temp() maps day-of-year -> temperature.
        # We evaluate it at a fine grid to produce a smooth background gradient.
        x_fine = np.linspace(germination_values[0], germination_values[-1], 500)
        flowering_intensity = np.array([floweringS(temp(x)) for x in x_fine])

        max_intensity = flowering_intensity.max()
        if max_intensity > 0:
            flowering_intensity_norm = flowering_intensity / max_intensity * 0.3
        else:
            flowering_intensity_norm = flowering_intensity

        # ── Publication-style rcParams ─────────────────────────────────────────────
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 13,
            'axes.linewidth': 1.2,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'xtick.direction': 'out',
            'ytick.direction': 'out',
            'xtick.major.width': 1.2,
            'ytick.major.width': 1.2,
            'xtick.major.size': 4,
            'ytick.major.size': 4,
            'legend.fontsize': 11,
            'legend.frameon': False,
            'lines.linewidth': 2.0,
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
        })

        # ── Build figure ───────────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

        # Determine y-axis limits across all data if not specified
        if ylim is None:
            all_vals = np.concatenate([r for panel in all_results for r in panel])
            ymax = np.ceil(np.nanmax(all_vals) * 20) / 20
            ylim = (0, max(ymax, 0.05))

        for col, (ax, panel_result, bj_label, panel_letter) in enumerate(
                zip(axes, all_results, bj_labels, panel_labels)):

            # ── Flowering background shading ───────────────────────────────────────
            # Draw thin vertical spans whose alpha encodes floweringS(temp(x)).
            # Using a fine grid avoids visible banding in the gradient.
            for i in range(len(x_fine) - 1):
                alpha = flowering_intensity_norm[i]
                if alpha > 1e-3:
                    ax.axvspan(x_fine[i], x_fine[i + 1],
                               color='dimgray', alpha=alpha,
                               linewidth=0, zorder=1)

            # ── Peak-flowering dotted vertical line ────────────────────────────────
            ax.axvline(x=182.5, color='dimgray', linestyle=':', linewidth=1.4,
                       zorder=2)

            # ── Data lines ────────────────────────────────────────────────────────
            for sweep_results, label, color in zip(panel_result, maturity_labels, colors):
                ax.plot(germination_values, sweep_results,
                        label=label, color=color, linewidth=2.0, zorder=3)

            # Panel letter in upper-left corner
            ax.text(0.04, 1.05, panel_letter,
                    transform=ax.transAxes,
                    fontsize=14, fontweight='bold',
                    va='top', ha='left')

            # Bj subtitle below panel letter
            ax.text(0.5, 1.04, bj_label,
                    transform=ax.transAxes,
                    fontsize=16, ha='center', va='bottom')

            if col == 0:
                ax.annotate(
                    'Peak flowering',
                    xy=(182.5, ylim[1] * 0.92),           # arrow tip — just below top
                    xytext=(182.5 + 60, ylim[1] * 0.92),  # label to the right
                    fontsize=14,
                    color='dimgray',
                    ha='left',
                    va='center',
                    arrowprops=dict(
                        arrowstyle='->', 
                        color='dimgray',
                        lw=1.2,
                    ),
                )

            ax.set_xlim(germination_values[0], germination_values[-1])
            ax.set_ylim(ylim)
            # ax.set_xlabel('Germination Time (day of year)', fontsize=14)
            ax.xaxis.set_major_locator(plt.MultipleLocator(60))
            ax.xaxis.set_minor_locator(plt.MultipleLocator(20))
            ax.tick_params(which='minor', length=2.5, width=0.8)
            ax.grid(True, axis='y', linestyle=':', linewidth=0.7, alpha=0.5, zorder=0)

            if col == 0:
                ax.set_ylabel('Prevalence', fontsize=14)
            else:
                ax.tick_params(left=True)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        fig.supxlabel('Germination Time (day of year)', fontsize=14, y=0.02)
        # ── Shared legend ──────────────────────────────────────────────────────────
        handles, labels_leg = axes[0].get_legend_handles_labels()

        flowering_patch = mpatches.Patch(color='dimgray', alpha=0.20,
                                        label='Flowering season')

        fig.legend(handles + [flowering_patch],
                labels_leg + ['Flowering season'],
                loc='lower center',
                ncol=4,
                bbox_to_anchor=(0.5, -0.10),
                frameon=False,
                fontsize=11)

        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"\nFigure saved to: {save_path}")

        plt.show()

        # Store for downstream access
        self.fig1_germination_values = germination_values
        self.fig1_maturity_values = maturity_values
        self.fig1_bj_values = bj_values
        self.fig1_results = all_results
        self.fig1_peak_info = peak_info

        return all_results
    
    def plot_fig2(self, infected_time_values, maturity_values, germination_time_values,
              maturity_labels=None, germination_labels=None,
              colors=None, figsize=(15, 5), num_processes=None,
              ylim=None, save_path=None, dpi=300, results_save_path=None,
              use_saved_embedding=False):
        """
        Publication-ready two-panel figure. Each panel sweeps infected_time on the
        x-axis with three maturation rates overlaid as lines. The two panels correspond
        to two different germination_time values. A vertical dashed line on each panel
        marks the germination pulse for that panel. Pink shading encodes germination
        intensity and gray shading encodes susceptible-flowering intensity across the
        x-axis.

        After the sweeps complete, the peak (max-prevalence) infected_time day of each
        curve is printed along with its mean/stdev across curves (see compute_curve_peaks
        / print_peak_summary). If results_save_path is given, the curves and peak stats
        are also saved as a reloadable embedding (see save_figure_embedding /
        load_figure_embedding / summarize_peak_embeddings).

        Args:
            infected_time_values:     Array of infected_time values to sweep (x-axis, day of year).
            maturity_values:          List of exactly 3 maturity rates (per-day) for line overlay.
            germination_time_values:  List of exactly 2 germination_time values, one per panel.
            maturity_labels:          Optional list of 3 legend labels for maturity lines.
            germination_labels:       Optional list of 2 panel subtitle strings.
            colors:                   Optional list of 3 line colors for the maturity lines.
            figsize:                  Figure size tuple. Default (10, 5).
            ylim:                     Optional (ymin, ymax) applied to all panels.
            save_path:                If provided, saves figure to this path (e.g. 'fig2.pdf').
            dpi:                      Resolution for saved figure. Default 300.
            num_processes:            Parallel processes. Defaults to cores.
            results_save_path:        If provided, saves the sweep curves + peak stats to
                                       this .npz path (e.g. 'fig2_embedding.npz').
            use_saved_embedding:      If True and results_save_path exists with matching
                                       infected_time_values/labels, skip the sweep entirely
                                       and rebuild the figure from the cached embedding.
        """
        if len(maturity_values) != 3:
            raise ValueError("maturity_values must contain exactly 3 values")
        if len(germination_time_values) != 3:
            raise ValueError("germination_time_values must contain exactly 3 values")

        if colors is None:
            colors = ['#2563eb', '#dc2626', '#16a34a']
        if maturity_labels is None:
            maturity_labels = [f"Maturity = {v * 365:.1f} yr⁻¹" for v in maturity_values]
        if germination_labels is None:
            germination_labels = [f"Germination peak = day {int(g)}" for g in germination_time_values]
        if num_processes is None:
            num_processes = cores

        infected_time_values = np.asarray(infected_time_values)
        panel_labels = ['A', 'B', 'C']

        flat_labels = [f"{germ_label} | {mat_label}"
                        for germ_label in germination_labels
                        for mat_label in maturity_labels]

        # ── Run all sweeps (or reuse a cached embedding) ────────────────────────────
        all_results = None
        reused = False

        if use_saved_embedding:
            cached_curves = _load_cached_curves(results_save_path, infected_time_values, flat_labels)
            if cached_curves is not None:
                n_mat = len(maturity_values)
                all_results = [cached_curves[i * n_mat:(i + 1) * n_mat]
                                for i in range(len(germination_time_values))]
                reused = True
                print(f"Reusing cached embedding from: {results_save_path}")
            else:
                print("No matching cached embedding found; running sweeps.")

        if all_results is None:
            all_results = []

            for panel_idx, germ_time in enumerate(germination_time_values):
                panel_results = []

                for maturity_val in maturity_values:
                    print(f"\n--- Panel {panel_labels[panel_idx]}: "
                        f"germination_time={germ_time:.1f}, maturity={maturity_val * 365:.2f}/yr ---")

                    model_kwargs = {
                        'Bj': self.model.Bj,
                        'Bv': self.model.Bv,
                        'Bf': self.model.Bf,
                        'births': self.model.births,
                        'gamma': self.model.gamma,
                        'death': self.model.death,
                        'maturity': maturity_val,
                        'g': self.model.g,
                        'susceptible_time': self.model.susceptible_time,
                        'infected_time': self.model.infected_time,
                        'infected_time2': self.model.infected_time2,
                        'germination_time': germ_time,
                        'eval': self.model.eval,
                        'X_0': self.model.X_0,
                        't': self.model.t
                    }

                    args_list = [
                        (inf_val, None, 'infected_time', None, model_kwargs, i)
                        for i, inf_val in enumerate(infected_time_values)
                    ]

                    sweep_results = np.zeros(len(infected_time_values))
                    total = len(infected_time_values)
                    completed = 0
                    start_time = time.time()
                    last_update = 0

                    with Pool(processes=num_processes) as pool:
                        for index_info, result in pool.imap(run_single_simulation, args_list):
                            sweep_results[index_info] = result if result is not None else 0.0
                            completed += 1
                            progress = (completed / total) * 100
                            if progress >= last_update + 25 or completed == total:
                                elapsed = time.time() - start_time
                                rate = completed / elapsed if elapsed > 0 else 1
                                remaining = (total - completed) / rate
                                print(f"  {int(progress)}% ({completed}/{total}) — "
                                    f"{elapsed:.1f}s elapsed, ~{remaining:.1f}s remaining")
                                last_update = int(progress / 25) * 25

                    panel_results.append(sweep_results)

                all_results.append(panel_results)

        # ── Peak summary across all curves ──────────────────────────────────────────
        flat_curves = [curve for panel in all_results for curve in panel]
        peak_info = compute_curve_peaks(infected_time_values, flat_curves, flat_labels)
        print_peak_summary("Fig 2", peak_info, x_label='infected_time day')

        if results_save_path is not None and not reused:
            save_figure_embedding(
                results_save_path, fig_name='fig2', x_name='infected_time',
                x_values=infected_time_values, curves=flat_curves, labels=flat_labels,
                extra_metadata={'germination_time_values': list(map(float, germination_time_values)),
                                 'maturity_values': list(map(float, maturity_values))},
            )

        # ── Precompute shading intensities across the x-axis ──────────────────────
        x_fine = np.linspace(infected_time_values[0], infected_time_values[-1], 500)

        # Gray: susceptible flowering season
        flowering_intensity = np.array([floweringS(temp(x)) for x in x_fine])
        max_fl = flowering_intensity.max()
        flowering_alpha = (flowering_intensity / max_fl * 0.45) if max_fl > 0 else flowering_intensity


        # ── Publication-style rcParams ─────────────────────────────────────────────
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 13,
            'axes.linewidth': 1.2,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'xtick.direction': 'out',
            'ytick.direction': 'out',
            'xtick.major.width': 1.2,
            'ytick.major.width': 1.2,
            'xtick.major.size': 4,
            'ytick.major.size': 4,
            'legend.fontsize': 11,
            'legend.frameon': False,
            'lines.linewidth': 2.0,
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
        })

        # ── Build figure ───────────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)

        if ylim is None:
            all_vals = np.concatenate([r for panel in all_results for r in panel])
            ymax = np.ceil(np.nanmax(all_vals) * 20) / 20
            ylim = (0, max(ymax, 0.05))

        for col, (ax, panel_result, germ_time, germ_label, panel_letter) in enumerate(
                zip(axes, all_results, germination_time_values, germination_labels, panel_labels)):
            germ_offset = 182.5 - germ_time
            germination_intensity = np.array([germination(temp(x + germ_offset)) for x in x_fine])
            max_germ = germination_intensity.max()
            germination_alpha = (germination_intensity / max_germ * 0.45) if max_germ > 0 else germination_intensity

            # ── Flowering shading (gray) ───────────────────────────────────────────
            for i in range(len(x_fine) - 1):
                if flowering_alpha[i] > 1e-3:
                    ax.axvspan(x_fine[i], x_fine[i + 1],
                               color='dimgray', alpha=flowering_alpha[i],
                               linewidth=0, zorder=1)

            # ── Germination shading (pink) ─────────────────────────────────────────
            for i in range(len(x_fine) - 1):
                if germination_alpha[i] > 1e-3:
                    ax.axvspan(x_fine[i], x_fine[i + 1],
                               color='#e75480', alpha=germination_alpha[i],
                               linewidth=0, zorder=1)

            # ── Data lines ────────────────────────────────────────────────────────
            for sweep_results, label, color in zip(panel_result, maturity_labels, colors):
                ax.plot(infected_time_values, sweep_results,
                        label=label, color=color, linewidth=2.0, zorder=3)

            # ── Peak vertical lines ────────────────────────────────────
            ax.axvline(x=germ_time, color='dimgray', linewidth=1.3,
                       linestyle='--', zorder=4)
            ax.axvline(x=182.5, color='dimgray', linestyle='--', linewidth=1.3, zorder=4)

            # Arrow annotation on left panel only
            if col == 0:
                # Peak germination arrow
                ax.annotate(
                    'Peak germination',
                    xy=(germ_time, ylim[1] * 0.92),
                    xytext=(182.5 + (infected_time_values[-1] - infected_time_values[0]) * 0.15,
                            ylim[1] * 0.92),
                    fontsize=14, color='#e75480',
                    ha='left', va='center',
                    arrowprops=dict(arrowstyle='->', color='#e75480', lw=1.2)
                )
                # Peak flowering arrow — placed lower to avoid overlap
                ax.annotate(
                    'Peak flowering',
                    xy=(182.5, ylim[1] * 0.86),
                    xytext=(182.5 + (infected_time_values[-1] - infected_time_values[0]) * 0.15,
                            ylim[1] * 0.86),
                    fontsize=14, color='dimgray',
                    ha='left', va='center',
                    arrowprops=dict(arrowstyle='->', color='dimgray', lw=1.2)
                )

            # Panel letter
            ax.text(0.04, 1.05, panel_letter,
                    transform=ax.transAxes,
                    fontsize=14, fontweight='bold',
                    va='top', ha='left')

            # Panel subtitle
            ax.text(0.5, 1.04, germ_label,
                    transform=ax.transAxes,
                    fontsize=16, ha='center', va='bottom')

            ax.set_xlim(infected_time_values[0], infected_time_values[-1])
            ax.set_ylim(ylim)
            ax.xaxis.set_major_locator(plt.MultipleLocator(60))
            ax.xaxis.set_minor_locator(plt.MultipleLocator(20))
            ax.tick_params(which='minor', length=2.5, width=0.8)
            ax.grid(True, axis='y', linestyle=':', linewidth=0.7, alpha=0.5, zorder=0)

            if col == 0:
                ax.set_ylabel('Prevalence', fontsize=14)
            else:
                ax.tick_params(left=True)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # ── Single centered x-axis title ───────────────────────────────────────────
        fig.supxlabel('Infectious Flowering Time (day of year)', fontsize=14, y=0.02)

        maturity_handles = [plt.Line2D([0], [0], color=c, linewidth=2.0) for c in colors]

        germination_patch = mpatches.Patch(color='#e75480', alpha=0.45, label='Germination season')
        flowering_patch   = mpatches.Patch(color='dimgray',  alpha=0.45, label='Flowering season')

        fig.legend(
            maturity_handles + [germination_patch, flowering_patch],
            maturity_labels  + ['Germination season', 'Flowering season'],
            loc='lower center',
            ncol=5,
            bbox_to_anchor=(0.5, -0.10),
            frameon=False,
            fontsize=11
        )

        fig.tight_layout()
        fig.subplots_adjust(bottom=0.12)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"\nFigure saved to: {save_path}")

        plt.show()

        self.fig2_infected_time_values = infected_time_values
        self.fig2_maturity_values = maturity_values
        self.fig2_germination_time_values = germination_time_values
        self.fig2_results = all_results
        self.fig2_peak_info = peak_info

        return all_results

    def plot_fig3(self, germination_values=None, flowering_values=None,
              base_maturity=4 / 365, maturity_multipliers=(4, 8, 16, 32),
              figsize=(7, 6), save_path=None, dpi=300, num_processes=None):
        """
        Publication-ready single-panel figure plotting the Evolutionarily Stable Strategy (ESS)
        flowering time against germination time for four maturation rates (multiples of a base
        maturity rate). For each (maturity multiplier, germination_time) pair, a full raster of
        infected_time vs infected_time2 is run via _compute_ess() (the same approach used by
        plot_fig4), and the ESS is the column with the minimum column sum. A diagonal reference
        line (ESS = germination time) is included for interpretive reference.

        Args:
            germination_values:   Array of germination_time values to sweep (x-axis, day of
                                  year). Defaults to np.linspace(62.5, 302.5, 9).
            flowering_values:     Array of infected_time / infected_time2 values for the inner
                                  raster grid. Defaults to np.linspace(0, 365, 40).
            base_maturity:        Base maturity rate (per day). Default 4/365.
            maturity_multipliers: Multipliers of base_maturity plotted as separate lines.
                                  Default (4, 8, 16, 32), matching the "4×"..."32×" legend.
            figsize:    Figure size tuple. Default (7, 6) for a single square-ish panel.
            save_path:  If provided, saves figure to this path (e.g. 'fig3.pdf').
            dpi:        Resolution for saved figure. Default 300.
            num_processes: Parallel processes for the inner raster. Defaults to cores.
        """
        if germination_values is None:
            germination_values = np.linspace(62.5, 302.5, 9)
        if flowering_values is None:
            flowering_values = np.linspace(0, 365, 40)
        if num_processes is None:
            num_processes = cores

        germination_times = np.asarray(germination_values)
        flowering_values = np.asarray(flowering_values)

        # ── Run the ESS sweep for each maturity multiplier ──────────────────────────
        ess_data = {}
        for mult in maturity_multipliers:
            label = f"{mult}×"
            maturity_val = base_maturity * mult
            print(f"\n--- m = {label} base (maturity = {maturity_val * 365:.2f} yr⁻¹) ---")

            ess_vals = np.zeros(len(germination_times))
            for g_idx, germ_time in enumerate(germination_times):
                model_kwargs = {
                    'Bj': self.model.Bj,
                    'Bv': self.model.Bv,
                    'Bf': self.model.Bf,
                    'births': self.model.births,
                    'gamma': self.model.gamma,
                    'death': self.model.death,
                    'maturity': maturity_val,
                    'g': self.model.g,
                    'susceptible_time': self.model.susceptible_time,
                    'infected_time': self.model.infected_time,
                    'infected_time2': self.model.infected_time2,
                    'germination_time': germ_time,
                    'eval': self.model.eval,
                    'X_0': self.model.X_0,
                    't': self.model.t,
                }
                start = time.time()
                ess = _compute_ess(model_kwargs, flowering_values, num_processes)
                ess_vals[g_idx] = ess
                print(f"  germination_time = {germ_time:.1f}  "
                      f"({g_idx + 1}/{len(germination_times)})  "
                      f"ESS = {ess:.2f}  ({time.time() - start:.1f}s)")

            ess_data[label] = ess_vals

        # ── Peak summary across all maturity multipliers ────────────────────────────
        # For each multiplier, the "peak" is the germination_time at which the ESS
        # flowering time is highest.
        peak_info = compute_curve_peaks(germination_times, list(ess_data.values()), list(ess_data.keys()))
        print_peak_summary("Fig 3", peak_info, x_label='germination day (ESS peak)')
        self.fig3_peak_info = peak_info

        colors = ['#2563eb', '#dc2626', '#16a34a', '#f59e0b']

        # ── Publication-style rcParams ─────────────────────────────────────────────
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 11,
            'axes.linewidth': 1.2,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'xtick.direction': 'out',
            'ytick.direction': 'out',
            'xtick.major.width': 1.2,
            'ytick.major.width': 1.2,
            'xtick.major.size': 4,
            'ytick.major.size': 4,
            'legend.fontsize': 10,
            'legend.frameon': False,
            'lines.linewidth': 2.0,
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
        })

        # ── Build figure ───────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=figsize)

        # Diagonal reference line (ESS = germination time)
        diag = np.array([germination_times[0], germination_times[-1]])
        ax.plot(diag, diag,
                color='#888888',
                linewidth=1.2,
                linestyle='--',
                zorder=2,
                label='ESS = germination time')

        # ESS lines per maturity multiplier
        for (label, ess_vals), color in zip(ess_data.items(), colors):
            ax.plot(germination_times, ess_vals,
                    color=color,
                    linewidth=2.0,
                    marker='o',
                    markersize=5,
                    markeredgewidth=0.8,
                    markeredgecolor='white',
                    zorder=3,
                    label=f"$m$ = {label} base")

        # Axis limits with a little breathing room
        pad = 10
        axis_min = germination_times[0] - pad
        axis_max = germination_times[-1] + pad
        ax.set_xlim(axis_min, axis_max)
        ax.set_ylim(axis_min, axis_max)

        ax.set_xlabel('Germination Time (day of year)', fontsize=12)
        ax.set_ylabel('ESS Flowering Time (day of year)', fontsize=12)
        ax.set_title('Evolutionarily Stable Infectious Flowering Time\nvs Germination Timing',
                    fontsize=12, fontweight='bold', pad=10)

        ax.xaxis.set_major_locator(plt.MultipleLocator(60))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(20))
        ax.yaxis.set_major_locator(plt.MultipleLocator(60))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(20))
        ax.tick_params(which='minor', length=2.5, width=0.8)

        ax.grid(True, linestyle=':', linewidth=0.7, alpha=0.5, zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.legend(loc='upper left', fontsize=10, frameon=False)

        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"\nFigure saved to: {save_path}")

        plt.show()

        # Store for downstream access
        self.fig3_germination_times = germination_times
        self.fig3_ess_data = ess_data

        return ess_data

    def plot_fig4(self, germination_values, flowering_values,
                  model_configs, config_labels=None, colors=None,
                  figsize=(14, 10), num_processes=None,
                  save_path=None, dpi=300, results_save_path=None):
        """
        Publication-ready figure that computes and plots the ESS infectious flowering
        time as a function of germination time for multiple parameter sets (e.g. basic,
        Silene latifolia, Dianthus pavonius).

        For each (config, germination_time) pair a full raster of infected_time vs
        infected_time2 is run via _compute_ess(), and the ESS is the column with the
        minimum column sum. Results are cached in self.fig4_results and optionally saved
        to a .npz file so expensive runs are not repeated.

        The figure has two rows:
          • Top row  — one panel per config (individual ESS vs germination-time curves).
          • Bottom   — one comparison panel overlaying all configs on a single axis.

        Args:
            germination_values:  Array of germination_time values to sweep (x-axis,
                                 day of year).  e.g. np.linspace(62.5, 302.5, 9).
            flowering_values:    Array of infected_time / infected_time2 values for the
                                 raster grid.  e.g. np.linspace(0, 365, 30).
            model_configs:       List of dicts, each providing Model kwargs that override
                                 self.model defaults for one scenario.  Use {} to keep
                                 all defaults (the "basic" model).
            config_labels:       Optional list of string labels for each config.
            colors:              Optional list of hex/named line colors (one per config).
            figsize:             Figure size tuple.  Default (14, 10).
            num_processes:       Worker processes for the inner raster.  Defaults to cores.
            save_path:           If provided, saves the figure (e.g. 'fig4.pdf').
            dpi:                 Resolution for saved figure.  Default 300.
            results_save_path:   If provided, saves all ESS arrays to a .npz file at
                                 this path.  Reload with:
                                     data = np.load(path)
                                     ess  = data['SileneLatifolia']
        Returns:
            dict mapping each label to its np.array of ESS flowering-time values.
        """
        import matplotlib.gridspec as gridspec

        def _jsonable(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, (np.integer, np.floating)):
                return value.item()
            if isinstance(value, dict):
                return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
            if isinstance(value, (list, tuple)):
                return [_jsonable(v) for v in value]
            return value

        def _cache_key(config, label):
            payload = {
                'label': label,
                'config': _jsonable(config),
                'flowering_values': _jsonable(flowering_values),
                'model_defaults': _jsonable({name: getattr(self.model, name) for name in (
                    'Bj', 'Bv', 'Bf', 'births', 'gamma', 'death', 'maturity', 'g',
                    'susceptible_time', 'infected_time', 'infected_time2', 'eval',
                    'X_0', 't')})
            }
            encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'))
            return hashlib.sha256(encoded.encode()).hexdigest()

        n_configs = len(model_configs)

        if config_labels is None:
            config_labels = [f"Config {i + 1}" for i in range(n_configs)]
        if colors is None:
            palette = ['#2563eb', '#dc2626', '#16a34a', '#f59e0b', '#7c3aed']
            colors = palette[:n_configs]
        if num_processes is None:
            num_processes = cores

        germination_values = np.asarray(germination_values)
        flowering_values   = np.asarray(flowering_values)

        cached = {}
        cached_germination = None
        cached_meta = {}
        if results_save_path is not None:
            try:
                with np.load(results_save_path, allow_pickle=False) as data:
                    cached_germination = np.asarray(data['germination_values'])
                    cached = {key: np.asarray(data[key]).copy() for key in data.files
                              if key not in ('germination_values', 'metadata')}
                    if 'metadata' in data.files:
                        cached_meta = json.loads(str(np.asarray(data['metadata']).item()))
                print(f"\nLoaded checkpoint: {results_save_path}")
            except (FileNotFoundError, KeyError, ValueError, OSError, json.JSONDecodeError):
                pass

        expected_keys = [_cache_key(config, label)
                         for config, label in zip(model_configs, config_labels)]
        cache_compatible = True
        if cached_meta:
            cache_compatible = (
                cached_meta.get('germination_values') == _jsonable(germination_values.tolist()) and
                cached_meta.get('cache_keys') == expected_keys)
            print(f"Checkpoint configuration match: {'yes' if cache_compatible else 'no'}")
            if cache_compatible:
                completed = sum(
                    int(np.isfinite(cached.get(
                        label.replace(' ', '_').replace('.', '').replace('(', '').replace(')', ''),
                        np.array([]))).sum())
                    for label in config_labels)
                total = n_configs * len(germination_values)
                print(f"Checkpoint progress: {completed}/{total} points complete.")
            else:
                print("Checkpoint configuration does not match; starting a fresh sweep.")
                cached = {}
        elif cached:
            # Legacy files do not record the model horizon or raster configuration,
            # so their values must not be reused for a new run.
            print("Checkpoint configuration match: no (metadata missing).")
            print(f"Checkpoint progress: 0/{n_configs * len(germination_values)} reusable points.")
            print("Ignoring legacy cached points.")
            cached = {}
            cache_compatible = False

        def _cached_value(label, germ_time):
            safe_label = label.replace(' ', '_').replace('.', '').replace('(', '').replace(')', '')
            values = cached.get(label, cached.get(safe_label))
            if values is None or cached_germination is None:
                return None
            matches = np.flatnonzero(np.isclose(cached_germination, germ_time, rtol=0, atol=1e-12))
            if len(matches) == 0 or matches[0] >= len(values):
                return None
            value = float(values[matches[0]])
            return value if np.isfinite(value) else None

        def _save_checkpoint(all_ess, peak_summary=None):
            if results_save_path is None:
                return
            safe = {k.replace(' ', '_').replace('.', '').replace('(', '').replace(')', ''): v
                    for k, v in all_ess.items()}
            metadata = {'germination_values': _jsonable(germination_values.tolist()),
                        'cache_keys': expected_keys}
            extra_arrays = {}
            if peak_summary is not None:
                metadata['peak_summary'] = peak_summary
                # Also write the generic embedding arrays (x_values/curves/labels/peak_*)
                # so this file can be read by load_figure_embedding() /
                # summarize_peak_embeddings() alongside fig1/fig2 embeddings.
                extra_arrays = {
                    'x_values': germination_values,
                    'curves': np.asarray(list(all_ess.values()), dtype=float),
                    'labels': np.array(peak_summary['labels']),
                    'peak_days': np.asarray(peak_summary['peak_days'], dtype=float),
                    'peak_values': np.asarray(peak_summary['peak_values'], dtype=float),
                }
            # Write atomically so an interruption cannot corrupt the checkpoint.
            checkpoint_path = os.fspath(results_save_path)
            temp_path = checkpoint_path + '.tmp.npz'
            np.savez(temp_path, germination_values=germination_values,
                     metadata=np.array(json.dumps(metadata)), **safe, **extra_arrays)
            os.replace(temp_path, checkpoint_path)

        # ── Compute ESS for every (config, germination_time) pair ─────────────────
        all_ess = {}

        for cfg_idx, (config, label) in enumerate(zip(model_configs, config_labels)):
            print(f"\n{'='*60}")
            print(f"Config: {label}  ({cfg_idx + 1}/{n_configs})")
            print(f"{'='*60}")

            ess_values = np.full(len(germination_values), np.nan)
            all_ess[label] = ess_values

            for g_idx, germ_time in enumerate(germination_values):
                cached_ess = _cached_value(label, germ_time) if cache_compatible else None
                if cached_ess is not None:
                    ess_values[g_idx] = cached_ess
                    print(f"\n  germination_time = {germ_time:.1f}  "
                          f"({g_idx + 1}/{len(germination_values)}) — cached ESS = {cached_ess:.2f}")
                    continue
                print(f"\n  germination_time = {germ_time:.1f}  "
                      f"({g_idx + 1}/{len(germination_values)})")

                # Build kwargs: self.model defaults → config overrides → sweep value
                model_kwargs = {
                    'Bj':              self.model.Bj,
                    'Bv':              self.model.Bv,
                    'Bf':              self.model.Bf,
                    'births':          self.model.births,
                    'gamma':           self.model.gamma,
                    'death':           self.model.death,
                    'maturity':        self.model.maturity,
                    'g':               self.model.g,
                    'susceptible_time':  self.model.susceptible_time,
                    'infected_time':     self.model.infected_time,
                    'infected_time2':    self.model.infected_time2,
                    'germination_time':  germ_time,
                    'eval':            self.model.eval,
                    'X_0':             self.model.X_0,
                    't':               self.model.t,
                }
                model_kwargs.update(config)
                model_kwargs['germination_time'] = germ_time  # always the sweep value

                start = time.time()
                ess = _compute_ess(model_kwargs, flowering_values, num_processes)
                ess_values[g_idx] = ess
                print(f"    ESS = {ess:.2f}  ({time.time() - start:.1f}s)")

                # Persist this point immediately so interrupted runs resume cleanly.
                _save_checkpoint(all_ess)

        # ── Peak summary across all configs ─────────────────────────────────────────
        # For each config, the "peak" is the germination_time at which the ESS
        # flowering time is highest.
        peak_info = compute_curve_peaks(germination_values, list(all_ess.values()), config_labels)
        print_peak_summary("Fig 4", peak_info, x_label='germination day (ESS peak)')

        # ── Persist results ────────────────────────────────────────────────────────
        self.fig4_results = {
            'germination_values': germination_values,
            **all_ess,
        }
        self.fig4_peak_info = peak_info

        if results_save_path is not None:
            peak_summary = {
                'labels': peak_info['labels'],
                'peak_days': peak_info['peak_days'].tolist(),
                'peak_values': peak_info['peak_values'].tolist(),
                'mean_peak_day': peak_info['mean_peak_day'],
                'std_peak_day': peak_info['std_peak_day'],
                'var_peak_day': peak_info['var_peak_day'],
                'mean_peak_value': peak_info['mean_peak_value'],
                'std_peak_value': peak_info['std_peak_value'],
                'var_peak_value': peak_info['var_peak_value'],
            }
            _save_checkpoint(all_ess, peak_summary=peak_summary)
            print(f"\nResults saved to: {results_save_path}")

        # ── Publication-style rcParams ─────────────────────────────────────────────
        plt.rcParams.update({
            'font.family':          'sans-serif',
            'font.sans-serif':      ['Arial', 'Helvetica', 'DejaVu Sans'],
            'font.size':            11,
            'axes.labelsize':       12,
            'axes.titlesize':       11,
            'axes.linewidth':       1.2,
            'xtick.labelsize':      10,
            'ytick.labelsize':      10,
            'xtick.direction':      'out',
            'ytick.direction':      'out',
            'xtick.major.width':    1.2,
            'ytick.major.width':    1.2,
            'xtick.major.size':     4,
            'ytick.major.size':     4,
            'legend.fontsize':      10,
            'legend.frameon':       False,
            'lines.linewidth':      2.0,
            'pdf.fonttype':         42,
            'ps.fonttype':          42,
        })

        # ── Axis limits ────────────────────────────────────────────────────────────
        pad      = 10
        axis_min = float(germination_values[0])  - pad
        axis_max = float(germination_values[-1]) + pad
        diag     = np.array([germination_values[0], germination_values[-1]])

        panel_letters = [chr(ord('A') + i) for i in range(n_configs + 1)]

        # ── Build figure with GridSpec ─────────────────────────────────────────────
        fig = plt.figure(figsize=figsize)
        gs  = gridspec.GridSpec(2, n_configs, figure=fig,
                                hspace=0.50, wspace=0.28,
                                height_ratios=[1, 1])

        # ── Top row: individual panels ─────────────────────────────────────────────
        for col, (label, ess_vals, color, letter) in enumerate(
                zip(config_labels, all_ess.values(), colors, panel_letters[:-1])):

            ax = fig.add_subplot(gs[0, col])

            # Diagonal reference line
            ax.plot(diag, diag,
                    color='#888888', linewidth=1.2, linestyle='--',
                    zorder=2, label='ESS = germination time')

            # ESS curve
            ax.plot(germination_values, ess_vals,
                    color=color, linewidth=2.0,
                    marker='o', markersize=5,
                    markeredgewidth=0.8, markeredgecolor='white',
                    zorder=3)

            ax.set_xlim(axis_min, axis_max)
            ax.set_ylim(axis_min, axis_max)
            ax.xaxis.set_major_locator(plt.MultipleLocator(60))
            ax.xaxis.set_minor_locator(plt.MultipleLocator(20))
            ax.yaxis.set_major_locator(plt.MultipleLocator(60))
            ax.yaxis.set_minor_locator(plt.MultipleLocator(20))
            ax.tick_params(which='minor', length=2.5, width=0.8)
            ax.grid(True, linestyle=':', linewidth=0.7, alpha=0.5, zorder=0)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Panel letter (bold, upper-left)
            ax.text(0.04, 1.10, letter,
                    transform=ax.transAxes,
                    fontsize=14, fontweight='bold', va='top', ha='left')

            # Config label as italic subtitle
            ax.text(0.5, 1.08, label,
                    transform=ax.transAxes,
                    fontsize=11, ha='center', va='bottom', style='italic')

            ax.set_xlabel('Germination Time (day of year)', fontsize=11)
            if col == 0:
                ax.set_ylabel('ESS Flowering Time (day of year)', fontsize=12)

        # ── Bottom row: comparison panel (spans all columns) ──────────────────────
        ax_comp = fig.add_subplot(gs[1, :])

        ax_comp.plot(diag, diag,
                     color='#888888', linewidth=1.2, linestyle='--',
                     zorder=2, label='ESS = germination time')

        for label, ess_vals, color in zip(config_labels, all_ess.values(), colors):
            ax_comp.plot(germination_values, ess_vals,
                         color=color, linewidth=2.0,
                         marker='o', markersize=5,
                         markeredgewidth=0.8, markeredgecolor='white',
                         zorder=3, label=label)

        ax_comp.set_xlim(axis_min, axis_max)
        ax_comp.set_ylim(axis_min, axis_max)
        ax_comp.xaxis.set_major_locator(plt.MultipleLocator(60))
        ax_comp.xaxis.set_minor_locator(plt.MultipleLocator(20))
        ax_comp.yaxis.set_major_locator(plt.MultipleLocator(60))
        ax_comp.yaxis.set_minor_locator(plt.MultipleLocator(20))
        ax_comp.tick_params(which='minor', length=2.5, width=0.8)
        ax_comp.grid(True, linestyle=':', linewidth=0.7, alpha=0.5, zorder=0)
        ax_comp.spines['top'].set_visible(False)
        ax_comp.spines['right'].set_visible(False)

        ax_comp.text(0.04, 1.07, panel_letters[n_configs],
                     transform=ax_comp.transAxes,
                     fontsize=14, fontweight='bold', va='top', ha='left')
        ax_comp.text(0.5, 1.05, 'Comparison',
                     transform=ax_comp.transAxes,
                     fontsize=11, ha='center', va='bottom')

        ax_comp.set_xlabel('Germination Time (day of year)', fontsize=12)
        ax_comp.set_ylabel('ESS Flowering Time (day of year)', fontsize=12)
        ax_comp.legend(loc='upper left', fontsize=10, frameon=False)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"\nFigure saved to: {save_path}")

        plt.show()

        # ── Cache for downstream access ────────────────────────────────────────────
        self.fig4_germination_values = germination_values
        self.fig4_flowering_values   = flowering_values
        self.fig4_model_configs      = model_configs
        self.fig4_config_labels      = config_labels
        self.fig4_all_ess            = all_ess

        return all_ess