"""
Save this entire file as: model_functions.py

Then in your Jupyter notebook, simply import and use:

    from model_functions import Model, RasterModel, GraphModel
    import numpy as np
    
    param1_values = np.linspace(0, 365, 13)
    param2_values = np.linspace(0, 365, 13)
    
    model = Model()
    raster_model = RasterModel(model, 'infected_offset', 'germination_offset', 
                               param1_values, param2_values)
    
    import time
    start_time = time.time()
    raster_model.raster()
    end_time = time.time()
    
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    raster_model.plot_heatmap()
"""

import multiprocessing as mp
from multiprocessing import Pool
import numpy as np
from scipy.integrate import solve_ivp, simpson
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import math

# Automatically detect available cores
cores = mp.cpu_count()

# Helper functions for model
def floweringS(temp, steepness=5, threshold=25, b_max=2):
    return b_max / (1 + np.exp(-steepness * (temp - threshold)))

def vegetatingS(temp, steepness=5, threshold=25, b_max=2):
    return b_max - b_max / (1 + np.exp(-steepness * (temp - threshold)))

def floweringI(temp, steepness=5, threshold=25, b_max=2):
    return b_max / (1 + np.exp(-steepness * (temp - threshold)))

def vegetatingI(temp, steepness=5, threshold=25, b_max=2):
    return b_max - b_max / (1 + np.exp(-steepness * (temp - threshold)))

def germination(temp, steepness=5, threshold=30, b_max=1, b_min=0):
    return b_max / (1 + np.exp(-steepness * (temp - threshold))) + b_min

def temp(t, min=0, scale=30):
    t = t * (2 * np.pi / 365) - np.pi
    temp_val = ((np.cos(t) + 1) / 2)
    temp_val = temp_val * scale
    temp_val = temp_val + min
    return temp_val

def tempVector(t):
    temps = np.zeros(len(t))
    for i, t_point in enumerate(t):
        temps[i] = temp(t_point)
    return temps


# Worker function - MUST be at module level
def run_single_simulation(args):
    """Worker function to run a single simulation with given parameters."""
    param1_val, param2_val, param1_name, param2_name, model_kwargs = args
    
    # Create a new model instance with the parameters
    model = Model(**model_kwargs)
    setattr(model, param1_name, param1_val)
    setattr(model, param2_name, param2_val)
    
    # Run simulation and evaluate
    solution = model.run_sim()
    result = model.evaluate(solution)
    
    return result


# Model class represents an entire system
class Model():
    def __init__(self, **kwargs):
        self.X_0 = (5, 0, 0, 5, 0, 0, 0, 0, 0, 0)
        self.t = (0, 365 * 50)

        # CONSTANTS
        self.Bj = 0.2 / 365
        self.Bv = 0.025 / 365
        self.Bf = 0.025 / 365
        self.births = 4 / 365
        self.gamma = 0.0001
        self.death = .5 / 365
        self.maturity = 8 / 365 #orig 2/365
        self.g = 2 / 365

        self.infected_offset = 0
        self.infected_offset2 = 0
        self.germination_offset = 0

        self.eval = 6

        for key, value in kwargs.items():
            setattr(self, key, value)

    def run_sim(self):
        self.solution = solve_ivp(self.df, self.t, self.X_0, max_step=0.1)

        self.Sj = self.solution.y[0, :]
        self.Sv = self.solution.y[1, :]
        self.Sf = self.solution.y[2, :]
        self.Ij = self.solution.y[3, :]
        self.Iv = self.solution.y[4, :]
        self.If = self.solution.y[5, :]
        self.Sd = self.solution.y[6, :]
        self.Ij2 = self.solution.y[7, :]
        self.Iv2 = self.solution.y[8, :]
        self.If2 = self.solution.y[9, :]

        self.time_points = self.solution.t

        return self.solution

    def df(self, t, X):
        Sj, Sv, Sf, Ij, Iv, If, Sd, Ij2, Iv2, If2 = X
        tCS = temp(t)
        tCI = temp(t + self.infected_offset)
        tCI2 = temp(t + self.infected_offset2)
        tCG = temp(t + self.germination_offset)

        dSd = self.births * Sf - germination(tCG) * Sd - (self.death / 4) * Sd
        dSj = germination(tCG) * Sd - self.Bj * Sj * (If + If2) - self.gamma * Sj * (
                    Sv + Sf + Iv + If + Iv2 + If2) - self.maturity * Sj - self.death * Sj
        dSv = self.maturity * Sj - self.Bv * Sv * (If + If2) - floweringS(tCS) * Sv + vegetatingS(
            tCS) * Sf - self.death * Sv
        dSf = floweringS(tCS) * Sv - self.Bf * Sf * (If + If2) - vegetatingS(tCS) * Sf - self.death * Sf
        dIj = self.Bj * Sj * If - self.maturity * Ij - self.death * Ij
        dIv = self.maturity * Ij + self.Bv * Sv * If - floweringI(tCI) * Iv + vegetatingI(
            tCI) * If - self.death * Iv
        dIf = floweringI(tCI) * Iv + self.Bf * Sf * If - vegetatingI(tCI) * If - self.death * If
        dIj2 = self.Bj * Sj * If2 - self.maturity * Ij2 - self.death * Ij2
        dIv2 = self.maturity * Ij2 + self.Bv * Sv * If2 - floweringI(tCI2) * Iv2 + vegetatingI(
            tCI2) * If2 - self.death * Iv2
        dIf2 = floweringI(tCI2) * Iv2 + self.Bf * Sf * If2 - vegetatingI(tCI2) * If2 - self.death * If2

        return (dSj, dSv, dSf, dIj, dIv, dIf, dSd, dIj2, dIv2, dIf2)

    def evaluate(self, solution):
        class1_rows = [4, 5, 3]
        class2_rows = [1, 2, 0]

        class1 = solution.y[class1_rows, :].sum(axis=0)
        class2 = solution.y[class2_rows, :].sum(axis=0)

        max1, min1 = -math.inf, math.inf
        max2, min2 = -math.inf, math.inf

        for temp_t in range(int(len(class1) * 0.8), int(len(class1))):
            if class1[temp_t] > max1:
                max1 = class1[temp_t]
            elif class1[temp_t] < min1:
                min1 = class1[temp_t]
            if class2[temp_t] > max2:
                max2 = class2[temp_t]
            elif class2[temp_t] < min2:
                min2 = class2[temp_t]

        match self.eval:
            case 0:
                return simpson(class1, x=solution.t)
            case 1:
                return simpson(class2, x=solution.t)
            case 2:
                return (
                        simpson(class1, x=solution.t)
                        / (
                                simpson(class1, x=solution.t)
                                + simpson(class2, x=solution.t)
                        )
                )
            case 3:
                return (max1 + min1) / 2
            case 4:
                return (max2 + min2) / 2
            case 5:
                return ((max1 + min1) / 2) - ((max2 + min2) / 2)
            case 6:
                return ((max1 + min1) / 2) / (((max1 + min1) / 2) + ((max2 + min2) / 2))
            case _:
                return 0


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

        # --- FIGURE 1 ---
        fig, populationGraph = plt.subplots(figsize=(10, 10))
        populationGraph.set_title('Population Levels (SI)')
        populationGraph.plot(time_points, total_S, label='Susceptibles', color=self.colSv)
        populationGraph.plot(time_points, total_I1, label='Infected', color=self.colIv)
        populationGraph.set_xlabel('Time (years)')
        populationGraph.set_ylabel('Abundance')
        populationGraph.set_ylim(0, 35)
        populationGraph.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 365:.1f}"))
        populationGraph.legend()

        # --- Figure 2: Custom Growth Rate Functions ---
        fig, custom2 = plt.subplots(figsize=(10, 10))
        custom2.set_title('Custom Growth Rate Functions')
        custom2.plot(time_points, germination(tempVector(time_points + self.model.germination_offset)),
                     label='Germination')
        custom2.plot(time_points, floweringS(tempVector(time_points)), label='Susceptible Flowering Rate',
                     color=self.colSf)
        custom2.plot(time_points, floweringI(tempVector(time_points + self.model.infected_offset)),
                     label='Infected Flowering Rate', color=self.colIf)
        custom2.plot(time_points, floweringI(tempVector(time_points + self.model.infected_offset2)),
                     label='Infected Flowering Rate2', color=self.colIf2)
        custom2.set_xlabel('time')
        custom2.set_ylabel('transition rate')
        custom2.set_xlim(0, 365 * 2)
        custom2.legend()

        # --- Figure 3: Genotype Frequency Over Time ---
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


# Raster class creates a graph to compare variable values
class RasterModel:
    def __init__(self, model, param1_name, param2_name, param1_values, param2_values):
        self.model = model
        self.param1_name = param1_name
        self.param2_name = param2_name
        self.param1_values = param1_values
        self.param2_values = param2_values
        self.heatmap_data = np.zeros((len(param1_values), len(param2_values)))

    def raster(self, num_processes=None):
        """
        Parallel version of raster using multiprocessing.
        
        Args:
            num_processes: Number of processes to use. If None, uses all available cores.
        """
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
            'infected_offset': self.model.infected_offset,
            'infected_offset2': self.model.infected_offset2,
            'germination_offset': self.model.germination_offset,
            'eval': self.model.eval,
            'X_0': self.model.X_0,
            't': self.model.t
        }
        
        # Prepare arguments for all simulations
        args_list = []
        for i, param1_val in enumerate(self.param1_values):
            for j, param2_val in enumerate(self.param2_values):
                args_list.append((param1_val, param2_val, self.param1_name, 
                                self.param2_name, model_kwargs))
        
        # Run simulations in parallel with progress tracking
        max_eval = -math.inf
        max_p1 = 0
        max_p2 = 0
        
        import time
        start_time = time.time()
        
        with Pool(processes=num_processes) as pool:
            # Use imap_unordered for progress tracking
            results = []
            completed = 0
            last_update = 0
            update_interval = 2  # Update every 2%
            
            print(f"Progress: 0% complete")
            for result in pool.imap_unordered(run_single_simulation, args_list):
                results.append(result)
                completed += 1
                
                # Calculate progress percentage
                progress = (completed / total_iterations) * 100
                
                # Print update every 10%
                if progress >= last_update + update_interval or completed == total_iterations:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    remaining = (total_iterations - completed) / rate if rate > 0 else 0
                    
                    print(f"Progress: {int(progress)}% complete "
                          f"({completed}/{total_iterations}) - "
                          f"Elapsed: {elapsed:.1f}s - "
                          f"Est. remaining: {remaining:.1f}s")
                    
                    last_update = int(progress / update_interval) * update_interval
        
        # Fill in the heatmap data (results are unordered, so we need to recompute)
        result_idx = 0
        for i, param1_val in enumerate(self.param1_values):
            for j, param2_val in enumerate(self.param2_values):
                cur_eval = results[result_idx]
                self.heatmap_data[i, j] = cur_eval
                
                if cur_eval > max_eval:
                    max_eval = cur_eval
                    max_p1 = param1_val
                    max_p2 = param2_val
                
                result_idx += 1
        
        total_time = time.time() - start_time
        print(f"\nRaster process complete in {total_time:.1f}s!")
        print(f"Max value: {max_eval}")
        print(f"At {self.param1_name} = {max_p1} and {self.param2_name} = {max_p2}")
        
        return self.heatmap_data

    def plot_heatmap(self):
        plt.figure(figsize=(8, 6))
        plt.imshow(self.heatmap_data, 
                   extent=[min(self.param2_values), max(self.param2_values),
                          min(self.param1_values), max(self.param1_values)],
                   origin="lower", aspect='auto', cmap="viridis")
        plt.colorbar(label="Pathogen Prevalence")
        plt.xlabel("Germination Time")
        plt.ylabel("Infection Flowering Time")
        plt.title(f"Pathogen Prevalence Across Varied Seasonal Peaks")
        plt.show()


# Test function if running directly
if __name__ == '__main__':
    print(f"Detected {cores} CPU cores")
    
    param1_values = np.linspace(0, 365, 13)
    param2_values = np.linspace(0, 365, 13)
    
    model = Model()
    raster_model = RasterModel(model, 'infected_offset', 'germination_offset', 
                               param1_values, param2_values)
    
    import time
    start_time = time.time()
    raster_model.raster()
    end_time = time.time()
    
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
    
    raster_model.plot_heatmap()