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
            print(f"Skipping {param1_name}={param1_val}, {param2_name}={param2_val} (will use symmetry)")
            return (index_info, None)

    solution = model.run_sim()
    model.eval = 6
    result = model.evaluate(solution)

    print(f"Completed simulation for {param1_name}={param1_val}, {param2_name}={param2_val} with result={result}")
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
            update_interval = 5  # Update every 5%
            
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
        print(f"Max value: {max_eval}")
        print(f"At {param_name} = {max_param}")
        
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
            update_interval = 2  # Update every 2%
            
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
        
        total_time = time.time() - start_time
        print(f"\nRaster process complete in {total_time:.1f}s!")
        print(f"Max value: {max_eval}")
        print(f"At {self.param1_name} = {max_p1} and {self.param2_name} = {max_p2}")
        
        return self.heatmap_data

    def plot_heatmap(self):
        if self.param1_values is None or self.param2_values is None:
            raise ValueError("Both parameters must be set for heatmap. Use plot_1d() for 1D sweeps.")
        
        plt.figure(figsize=(8, 6))
        plt.imshow(
            self.heatmap_data,
            extent=[min(self.param2_values), max(self.param2_values),
                    min(self.param1_values), max(self.param1_values)],
            origin="lower",
            aspect="auto",
            cmap="viridis",
            # vmin=0.0,
            # vmax=1.0,   # force colormap range 0–1
        )
        plt.colorbar(label="Pathogen Prevalence")
        plt.xlabel("Infection Genotype 2 Flowering Time")
        plt.ylabel("Infection Genotype 1 Flowering Time")
        plt.title("Pathogen Prevalence Across Varied Seasonal Peaks")
        plt.show()

