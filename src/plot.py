# models_plotting.py
import math
import time
import numpy as np
from multiprocessing import Pool

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from timing import (
    cores, floweringS, vegetatingS, floweringI, vegetatingI,
    germination, tempVector
)
from model import Model

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

        fig, populationGraph = plt.subplots(figsize=(10, 10))
        populationGraph.set_title('Population Levels (SI)')
        populationGraph.plot(time_points, total_S, label='Susceptibles', color=self.colSv)
        populationGraph.plot(time_points, total_I1, label='Infected1', color=self.colIv)
        populationGraph.plot(time_points, total_I2, label='Infected2', color=self.colIv2)
        populationGraph.set_xlabel('Time (years)')
        populationGraph.set_ylabel('Abundance')
        populationGraph.set_ylim(0, 35)
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
        prevalenceGraph.fill_between(time_points, 0, I1_percent, color=self.colIf, alpha=0.7, label='Infected 1')
        prevalenceGraph.fill_between(time_points, I1_percent, 100, color=self.colIf2, alpha=0.7, label='Infected 2')
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
              colors=None, figsize=(15, 5), num_processes=None,
              ylim=None, save_path=None, dpi=300):
        """
        Publication-ready three-panel figure. Each panel sweeps germination_time on the
        x-axis with three maturation rates overlaid as lines. The three panels correspond
        to three different transmission rates (Bj).

        Args:
            germination_values: Array of germination_time values to sweep (x-axis).
            maturity_values:    List of exactly 3 maturity rates (per-day) for the line overlay.
            bj_values:          List of exactly 3 Bj values, one per panel.
            maturity_labels:    Optional list of 3 legend labels for maturity lines.
            bj_labels:          Optional list of 3 panel subtitle strings (e.g. ["β_J = 0.1", ...]).
            colors:             Optional list of 3 line colors for the maturity lines.
            figsize:            Figure size tuple. Default (15, 5) suits 3 wide panels.
            ylim:               Optional (ymin, ymax) tuple applied to all panels.
            save_path:          If provided, saves figure to this path (e.g. 'fig1.pdf').
            dpi:                Resolution for saved figure. Default 300.
            num_processes:      Parallel processes. Defaults to cores.
        """
        if len(maturity_values) != 3:
            raise ValueError("maturity_values must contain exactly 3 values")
        if len(bj_values) != 3:
            raise ValueError("bj_values must contain exactly 3 values")

        if colors is None:
            colors = ['#2563eb', '#dc2626', '#16a34a']
        if maturity_labels is None:
            maturity_labels = [f"Maturity = {v * 365:.1f} yr⁻¹" for v in maturity_values]
        if bj_labels is None:
            bj_labels = [f"$\\beta_J$ = {bj * 365:.2f} yr⁻¹" for bj in bj_values]
        if num_processes is None:
            num_processes = cores

        germination_values = np.asarray(germination_values)
        panel_labels = ['A', 'B', 'C']

        # ── Run all sweeps ─────────────────────────────────────────────────────────
        # Shape: [n_panels, n_maturity_values, n_germination_values]
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
            'pdf.fonttype': 42,   # embeds fonts properly for journals
            'ps.fonttype': 42,
        })

        # ── Build figure ───────────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)

        # Determine y-axis limits across all data if not specified
        if ylim is None:
            all_vals = np.concatenate([r for panel in all_results for r in panel])
            ymax = np.ceil(np.nanmax(all_vals) * 20) / 20   # round up to nearest 0.05
            ylim = (0, max(ymax, 0.05))

        for col, (ax, panel_result, bj_label, panel_letter) in enumerate(
                zip(axes, all_results, bj_labels, panel_labels)):

            for sweep_results, label, color in zip(panel_result, maturity_labels, colors):
                ax.plot(germination_values, sweep_results,
                        label=label, color=color, linewidth=2.0, zorder=3)

            # Panel letter in upper-left corner
            ax.text(0.04, 0.96, panel_letter,
                    transform=ax.transAxes,
                    fontsize=13, fontweight='bold',
                    va='top', ha='left')

            # Bj subtitle below panel letter
            ax.text(0.5, 1.04, bj_label,
                    transform=ax.transAxes,
                    fontsize=11, ha='center', va='bottom')

            ax.set_xlim(germination_values[0], germination_values[-1])
            ax.set_ylim(ylim)
            ax.set_xlabel('Germination Time (day of year)', fontsize=12)
            ax.xaxis.set_major_locator(plt.MultipleLocator(60))
            ax.xaxis.set_minor_locator(plt.MultipleLocator(20))
            ax.tick_params(which='minor', length=2.5, width=0.8)
            ax.grid(True, axis='y', linestyle=':', linewidth=0.7, alpha=0.5, zorder=0)

            # Only leftmost panel gets a y-axis label
            if col == 0:
                ax.set_ylabel('Equilibrium Prevalence', fontsize=12)
            else:
                ax.tick_params(left=True)   # keep ticks, sharey handles labels

            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # Single shared legend below the figure
        handles, labels_leg = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels_leg,
                loc='lower center',
                ncol=3,
                bbox_to_anchor=(0.5, -0.08),
                frameon=False,
                fontsize=11)

        fig.suptitle('Equilibrium Prevalence vs Germination Timing',
                    fontsize=13, fontweight='bold', y=1.02)

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

        return all_results
    
    def plot_fig2(self, infected_time_values, maturity_values, germination_time_values,
              maturity_labels=None, germination_labels=None,
              colors=None, figsize=(15, 5), num_processes=None,
              ylim=None, save_path=None, dpi=300):
        """
        Publication-ready three-panel figure. Each panel sweeps infected_time on the
        x-axis with three maturation rates overlaid as lines. The three panels correspond
        to three different germination_time values. A vertical dashed line on each panel
        marks the germination pulse for that panel.

        Args:
            infected_time_values:     Array of infected_time values to sweep (x-axis, day of year).
            maturity_values:          List of exactly 3 maturity rates (per-day) for line overlay.
            germination_time_values:  List of exactly 3 germination_time values, one per panel.
            maturity_labels:          Optional list of 3 legend labels for maturity lines.
            germination_labels:       Optional list of 3 panel subtitle strings.
            colors:                   Optional list of 3 line colors for the maturity lines.
            figsize:                  Figure size tuple. Default (15, 5).
            ylim:                     Optional (ymin, ymax) applied to all panels.
            save_path:                If provided, saves figure to this path (e.g. 'fig2.pdf').
            dpi:                      Resolution for saved figure. Default 300.
            num_processes:            Parallel processes. Defaults to cores.
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

        # ── Run all sweeps ─────────────────────────────────────────────────────────
        # Shape: [n_panels, n_maturity_values, n_infected_time_values]
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
                    'germination_time': germ_time,          # <-- fixed per panel
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
        fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)

        # Auto y-limits from data if not specified
        if ylim is None:
            all_vals = np.concatenate([r for panel in all_results for r in panel])
            ymax = np.ceil(np.nanmax(all_vals) * 20) / 20
            ylim = (0, max(ymax, 0.05))

        for col, (ax, panel_result, germ_time, germ_label, panel_letter) in enumerate(
                zip(axes, all_results, germination_time_values, germination_labels, panel_labels)):

            # Data lines
            for sweep_results, label, color in zip(panel_result, maturity_labels, colors):
                ax.plot(infected_time_values, sweep_results,
                        label=label, color=color, linewidth=2.0, zorder=3)

            # Vertical line marking the germination pulse for this panel
            ax.axvline(x=germ_time,
                    color='#444444',
                    linewidth=1.3,
                    linestyle='--',
                    zorder=4,
                    label=f'Germination pulse (day {int(germ_time)})')

            # Panel letter
            ax.text(0.04, 0.96, panel_letter,
                    transform=ax.transAxes,
                    fontsize=13, fontweight='bold',
                    va='top', ha='left')

            # Panel subtitle (germination time)
            ax.text(0.5, 1.04, germ_label,
                    transform=ax.transAxes,
                    fontsize=11, ha='center', va='bottom')

            ax.set_xlim(infected_time_values[0], infected_time_values[-1])
            ax.set_ylim(ylim)
            ax.set_xlabel('Infectious Flowering Time (day of year)', fontsize=12)
            ax.xaxis.set_major_locator(plt.MultipleLocator(60))
            ax.xaxis.set_minor_locator(plt.MultipleLocator(20))
            ax.tick_params(which='minor', length=2.5, width=0.8)
            ax.grid(True, axis='y', linestyle=':', linewidth=0.7, alpha=0.5, zorder=0)

            if col == 0:
                ax.set_ylabel('Equilibrium Prevalence', fontsize=12)
            else:
                ax.tick_params(left=True)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # Arrow label only on leftmost panel
        axes[0].annotate('Germ.\npulse',
            xy=(germination_time_values[0], ylim[1]),
            xytext=(germination_time_values[0] + (infected_time_values[-1] - infected_time_values[0]) * 0.04,
                    ylim[1] * 0.92),
            fontsize=8.5,
            color='#444444',
            ha='left',
            va='top',
            arrowprops=dict(arrowstyle='->', color='#444444', lw=1.0))
    
        # Shared legend — maturity lines only (vertical line label handled by annotation)
        maturity_handles, maturity_leg_labels = [], []
        for sweep_results, label, color in zip(all_results[0], maturity_labels, colors):
            maturity_handles.append(plt.Line2D([0], [0], color=color, linewidth=2.0))
            maturity_leg_labels.append(label)

        # Add germination line to legend once
        maturity_handles.append(plt.Line2D([0], [0], color='#444444', linewidth=1.3, linestyle='--'))
        maturity_leg_labels.append('Germination pulse')

        fig.legend(maturity_handles, maturity_leg_labels,
                loc='lower center',
                ncol=4,
                bbox_to_anchor=(0.5, -0.08),
                frameon=False,
                fontsize=11)

        fig.suptitle('Equilibrium Prevalence vs Infectious Flowering Timing',
                    fontsize=13, fontweight='bold', y=1.02)

        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"\nFigure saved to: {save_path}")

        plt.show()

        # Store for downstream access
        self.fig2_infected_time_values = infected_time_values
        self.fig2_maturity_values = maturity_values
        self.fig2_germination_time_values = germination_time_values
        self.fig2_results = all_results

        return all_results

    def plot_fig3(self, figsize=(7, 6), save_path=None, dpi=300):
        """
        Publication-ready single-panel figure plotting the Evolutionarily Stable Strategy (ESS)
        flowering time against germination time for four maturation rates (multiples of m=4/365).
        Data are hardcoded from simulation output. A diagonal reference line (ESS = germination
        time) is included for interpretive reference.

        Args:
            figsize:    Figure size tuple. Default (7, 6) for a single square-ish panel.
            save_path:  If provided, saves figure to this path (e.g. 'fig3.pdf').
            dpi:        Resolution for saved figure. Default 300.
        """
        # ── Hardcoded simulation output ────────────────────────────────────────────
        # Base maturity m = 4/365; columns are multiples of that base
        germination_times = np.array([62.5, 92.5, 122.5, 152.5, 182.5, 212.5, 242.5, 272.5, 302.5])

        ess_data = {
            '16×': np.array([93.5897436, 121.6666667, 154.4230769, 187.1794872,
                            215.2564103, 243.3333333, 271.4102564, 304.1666667, 336.9230769]),
            '8×':  np.array([93.5897436, 121.6666667, 154.4230769, 187.1794872,
                            215.2564103, 243.3333333, 276.0897436, 304.1666667, 336.9230769]),
            '4×':  np.array([93.5897436, 126.3461538, 154.4230769, 187.1794872,
                            215.2564103, 248.0128205, 276.0897436, 304.1666667, 336.9230769]),
            '2×':  np.array([98.2692308, 126.3461538, 159.1025641, 187.1794872,
                            219.9358974, 248.0128205, 276.0897436, 308.8461538, 336.9230769]),
        }

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