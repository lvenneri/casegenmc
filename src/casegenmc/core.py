import copy

import numpy as np
import pandas as pd
from casegenmc.util import timer, clean_fld_name
from os.path import join as pjoin
from scipy.stats import uniform, norm, lognorm, sobol_indices
from tqdm import tqdm

# from scipy.stats import sobol_indices

from casegenmc.plotting_base import *
import itertools
import casegenmc.tex_plots as tex_plots 

PARALLEL = False

try:
    import ray
except ImportError:
    ray = None  # Flag that Ray is not available

def init_casegenmc( setup_tex=False, texfonts=True, fontsize=8, figsize=(6, 6)
):


    tex_plots.TEX_PLOTS = setup_tex

    if setup_tex:
        try:
            tex_plots.setup_tex_plots(texfonts=texfonts, fontsize=fontsize, figsize=figsize)
        except:
            print("tex fonts not found, skipping")

    if fontsize:
        plt.rcParams.update({"font.size": fontsize})
    if figsize:
        plt.rcParams.update({"figure.figsize": figsize})



def check_input_valid(base_inputs, add_inputs):
    for k, v in add_inputs.items():
        if k not in base_inputs:
            raise ValueError(
                f"Parameter {k} not in base_inputs. Check the specific inputs fed."
            )

        if isinstance(v, (list, np.ndarray)):
            if not (
                isinstance(v[0], type(base_inputs[k]))
                or (
                    isinstance(v[0], (int, float))
                    and isinstance(base_inputs[k], (int, float))
                )
            ):
                raise TypeError(
                    f"Type mismatch for parameter {k}. Expected items of type {type(base_inputs[k])}, got item of type {type(v[0])}."
                )
        elif not (
            isinstance(v, type(base_inputs[k]))
            or (
                isinstance(v, (int, float)) and isinstance(base_inputs[k], (int, float))
            )
        ):
            raise TypeError(
                f"Type mismatch for parameter {k}. Expected {type(base_inputs[k])}, got {type(v)}."
            )

    # check there are no repeats
    if len(add_inputs.keys()) != len(set(add_inputs.keys())):
        raise ValueError("There are repeats in the parameter space")

    return


def process_input_stack(input_stack0, default_unc_type="normal", default_unc_frac=0):
    """
    Processes the input stack to ensure all parameters are correctly formatted for analysis.

    This function iterates through each key-value pair in the input stack. If the value is a simple data type (int, float, str, bool),
    it converts it into a dictionary with default settings. For dictionary type values, it ensures all necessary keys are present
    and correctly formatted. It sets 'unc_type' to 'choice' for categorical data and handles the probability distributions for options.
    It also converts 'unc_frac' to 'unc' where necessary.

    Parameters:
    - input_stack (dict): The input stack where each key is a parameter name and each value is either a simple data type or a dictionary
      specifying details about the parameter. See example.
    - default_unc_type (str, optional): The default type of uncertainty if none is specified. Defaults to 'normal'.
    - default_unc_frac (float, optional): The default fraction of the mean to use as the standard deviation if 'unc' is not defined. Defaults to 0.

    Returns:
    - dict: The processed input stack with all parameters correctly formatted as dictionaries.
    """
    input_stack = copy.deepcopy(input_stack0)
    for key, value in input_stack.items():
        if isinstance(value, (int, float, str, bool)):
            input_stack[key] = {
                "mean": value,
                "unc": 0,
                "options": [value],
                # "unc_type": "choice" if not isinstance(value, (int, float)) else "normal",
                "type": "options",
            }

        value = input_stack[key]

        if isinstance(value, dict):

            if "options" in value or isinstance(value["mean"],str):
                if "unc_type" not in value:
                    value["unc_type"] = "choice"
                elif value["unc_type"] != "choice":
                    raise ValueError(f"Uncertainty type {value['unc_type']} is not compatible with options for key '{key}'")
                value["bounds"] = None
                value["type"] = "options"
                value["unc_frac"] = None
                value["unc"] = None

                if "range" in value:
                    if "range" == "all":
                        value["range"] = value["options"]
                else:
                    value["range"] = [value["mean"]] # so no choice

                if "options" not in value:
                    value["options"] = value["mean"]

                # check that range is in options
                if not all(v in value["options"] for v in value["range"]):
                    raise ValueError(
                        f"Range {value['range']} is not in options {value['options']} for key '{key}'"
                    )

                if "options" not in value:
                    value["options"] = [value["mean"]]

                if "prob" in value:
                    if not (len(value["options"]) == len(value["prob"])):
                        raise ValueError(
                            f"Length of 'prob' ({len(value['prob'])}) must match length of 'options' ({len(value['options'])}) for key '{key}'"
                        )
                    # update prob for the selected range (which is subset of options)
                    range_i = [value["options"].index(v) for v in value["range"]]
                    value["prob"] = np.array(value["prob"])[range_i]
                    value["prob"] = value["prob"] / value["prob"].sum()
                else:
                    value["prob"] = np.ones(len(value["range"])) / len(value["range"])
            else:

                if "unc_frac" in value:
                    value["unc"] = None
                elif "unc" in value:
                    value["unc_frac"] = None
                else:
                    value["unc_frac"] = default_unc_frac

                if "range" not in value:
                    umag = value["unc_frac"] * value["mean"] if "unc_frac" in value else value["unc"]

                    value["range"] = [
                        value["mean"] - 3 * umag,
                        value["mean"] + 3 * umag,
                    ]
                else:

                    if "bounds" in value:
                        if not (
                            value["bounds"][0]
                            <= value["range"][0]
                            <= value["bounds"][1]
                        ):
                            raise ValueError(
                                f"Range {value['range']} is not in bounds {value['bounds']} for key '{key}'"
                            )

                if "unc_type" not in value:
                    if isinstance(value["mean"], (int, float)):
                        value["unc_type"] = default_unc_type

                if "bounds" not in value:
                    value["bounds"] = [0, 100 * value["mean"]]
                if "type" not in value:
                    value["type"] = type(value["mean"]).__name__

        # check that mean is in range and bounds
        if input_stack[key]["type"] == "float" or input_stack[key]["type"] == "int":
            if not (
                input_stack[key]["range"][0]
                <= input_stack[key]["mean"]
                <= input_stack[key]["range"][1]
            ):
                raise ValueError(
                    f"Mean {input_stack[key]['mean']} is not in range {input_stack[key]['range']} for key '{key}'"
                )
            if "bounds" in input_stack[key] and not (
                input_stack[key]["bounds"][0]
                <= input_stack[key]["mean"]
                <= input_stack[key]["bounds"][1]
            ):
                raise ValueError(
                    f"Mean {input_stack[key]['mean']} is not in bounds {input_stack[key]['bounds']} for key '{key}'"
                )

    return input_stack


def worker_task(index, case, model):
    """
    Standard function that returns a tuple: (index, result_dict).
    """
    return index, model(case)


def run_cases(inputs, model, output_stats=False, parallel=False, num_cpus=None, batch_size=None):
    """
    Robust run_cases that works even if Ray is not installed.
    """
    data_out_dir = "./data"
    if not os.path.exists(data_out_dir):
        os.makedirs(data_out_dir)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_file = pjoin(data_out_dir, f"outputs_{timestamp}.csv")

    start_time = time.time()

    # Normalize inputs
    if isinstance(inputs, pd.DataFrame):
        cases_list = inputs.to_dict('records')
        inputs_df = inputs.copy().reset_index(drop=True)
    elif isinstance(inputs, list):
        cases_list = inputs
        inputs_df = pd.DataFrame(inputs)
    else:
        raise ValueError("Inputs must be a list of dicts or a pandas DataFrame.")

    # --- 3. Parallel Logic Check ---
    if parallel:
        if ray is None:
            print("WARNING: 'parallel=True' requested but 'ray' is not installed.")
            print("Falling back to serial execution.")
            parallel = False
        else:
            # Ray is available, proceed with setup
            if not ray.is_initialized():
                ray.init(num_cpus=num_cpus)

            # --- DYNAMIC REMOTING ---
            # We convert the plain python function to a Ray remote function strictly at runtime
            remote_worker = ray.remote(worker_task)

            print(f"Launching {len(cases_list)} tasks on {num_cpus if num_cpus else 'all'} cores...")

            # Use the dynamic 'remote_worker' instead of the function name directly
            futures = [remote_worker.remote(i, case, model) for i, case in enumerate(cases_list)]

            if batch_size is None:
                batch_size = len(cases_list)

            header_written = False

            # Batch Loop
            while futures:
                done_futures, futures = ray.wait(futures, num_returns=min(batch_size, len(futures)))
                batch_results = ray.get(done_futures)

                batch_rows = []
                for idx, res in batch_results:
                    input_row = inputs_df.iloc[[idx]].to_dict('records')[0]
                    batch_rows.append({**input_row, **res})

                batch_df = pd.DataFrame(batch_rows)
                mode = 'a' if header_written else 'w'
                header = not header_written
                batch_df.to_csv(output_file, mode=mode, header=header, index=False)
                header_written = True

                print(f"Batch processed: {len(batch_rows)} items written.")
                del batch_results, batch_df, batch_rows

    # --- 4. Serial Fallback ---
    # This runs if parallel=False OR if Ray was missing
    if not parallel:
        print("Running in serial mode...")
        # We can still use batch writing in serial to save memory
        header_written = False
        buffer = []
        eff_batch_size = batch_size if batch_size else 1000  # Default buffer for serial

        for i, case in enumerate(cases_list):
            # Run the plain function directly
            _, res = worker_task(i, case, model)

            input_row = inputs_df.iloc[[i]].to_dict('records')[0]
            buffer.append({**input_row, **res})

            if len(buffer) >= eff_batch_size:
                batch_df = pd.DataFrame(buffer)
                mode = 'a' if header_written else 'w'
                header = not header_written
                batch_df.to_csv(output_file, mode=mode, header=header, index=False)
                header_written = True
                buffer = []  # Clear memory

        # Write remaining
        if buffer:
            batch_df = pd.DataFrame(buffer)
            mode = 'a' if header_written else 'w'
            header = not header_written
            batch_df.to_csv(output_file, mode=mode, header=header, index=False)

    print(f"--- Finished in {(time.time() - start_time):.2f}s ---")

    # --- 5. Return Logic ---
    # (Same as before: try to load file if it fits in memory, else return path)
    if output_stats:
        try:
            full_df = pd.read_csv(output_file)
            out_stats = calculate_stats(full_df)
        except MemoryError:
            print("Output too large for stats.")
            full_df = None
            out_stats = None
    else:
        # If user wanted batching, assume they might not want the huge DF back
        if batch_size and len(cases_list) > 10000:
            full_df = None
        else:
            full_df = pd.read_csv(output_file)
        out_stats = None

    return {"out": full_df, "out_stats": out_stats, "file_path": output_file}


def calculate_stats(df):
    # (Same helper as before)
    stats_dict = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            desc = df[col].describe()
            stats_dict[col] = {
                "mean": desc['mean'], "std": desc['std'],
                "min": desc['min'], "max": desc['max']
            }
    return pd.DataFrame.from_dict(stats_dict, orient='index')


def generate_combos(par_space, type="dict"):
    """
    Grid points for parameter space.

    Example:
    # par_space = {'radius': np.linspace(1, 100, 10),
    #              'thickness':["sph", "cyl"],
    #              }
    # print(generate_combos(par_space))

    :param par_space:
    :return:

    """
    parameter = []
    possible_vals = []

    for par, par_range in par_space.items():
        parameter.append(par)
        possible_vals.append(par_range)
    par_tests = list(itertools.product(*possible_vals))

    if type == "dict":
        # turn par_tests into a dictionary with one entry for each test
        combos_data = {}
        for i in range(len(par_tests)):
            combos_data[i] = {}
            for j in range(len(parameter)):
                combos_data[i][parameter[j]] = par_tests[i][j]

    else:
        
            par_tests_array = np.array(par_tests, dtype=object)

            combos_data = pd.DataFrame(
                data=par_tests_array,
                columns=parameter,
            )

    return combos_data


def generate_combos_rand(par_space, n=1000, o_vals=True):
    """

    The input 'par_space' is a dictionary of parameter names with either a tuple of (min, max) or a list of
    values to choose from. The function returns a dictionary containing 'n' randomly generated combinations
    of the parameters in the input dictionary.

    Parameters
    ----------
    par_space : dict
        A dictionary of parameter names with either a tuple of (min, max) or a list of values to choose from.
        Example:
        generate_combos_rand({'a': (1, 5), 'b': [2, 4, 6]},n=1000)

    n : int, optional, default: 1000
        The number of random combinations of parameters to generate.
    o_vals: use the original ranges of values and lists, give 0 to 1, and 0,1,2.. number of vlaues in list

    Returns
    -------
    par_space_ds : dict
        A dictionary containing 'n' randomly generated combinations of the parameters in the input dictionary.
        The keys are integers from 0 to n-1, and the values are dictionaries containing parameter names as keys
        and corresponding parameter values as values.
        Example:
            {0: {'a': 2.5, 'b': 4},
             1: {'a': 4.2, 'b': 2},
             ...}

    """

    if o_vals:
        par_space_ds = {}
        for i in range(n):
            par_space_ds[i] = {}
            for k, v in par_space.items():
                if isinstance(v, tuple):
                    # Generate random float within the original range
                    par_space_ds[i][k] = np.random.uniform(v[0], v[1])
                elif isinstance(v, list) or isinstance(v, np.ndarray):
                    # Randomly select an element from the list or array
                    par_space_ds[i][k] = np.random.choice(v)

    else:
        par_space_ds = []
        for i in range(n):
            par_space_ds_i = []
            for k, v in par_space.items():
                if isinstance(v, tuple):
                    # Generate random float between 0 and 1
                    par_space_ds_i.append(np.random.uniform(0, 1))
                elif isinstance(v, list) or isinstance(v, np.ndarray):
                    # Randomly select an index from the range of the list or array's length
                    par_space_ds_i.append(np.random.choice(range(len(v))))

            par_space_ds.append(par_space_ds_i)
        par_space_ds = np.vstack(par_space_ds)

    return par_space_ds


# @timer
def generate_samples(par_space0, type="unc", n=1000, par_to_sample=None, grid_n=None):
    """
    Generates samples from a parameter space.

    Parameters
    ----------
    par_space : dict
        Dictionary specifying the parameter space.
    type : str, optional
        Type of sampling to perform. Must be one of:
        - "unc": Samples based on the uncertainty type and range specified for each parameter in par_space. It will use the "unc_frac" to calculate the std of the distribution. If "unc_frac" is not specified, it will use "unc"
                 For parameters with "unc_type" specified, it will sample from the corresponding distribution
                 (e.g., normal with mean and standard deviation, uniform over range, etc.).
                 For parameters with "options" specified, it will randomly select from the options according to their probabilities.
        - "uniform": Samples uniformly over the range specified for each parameter, ignoring any uncertainty or distribution.
        - "grid": Generates a regular grid of samples over the range specified for each parameter.
        - "extremes": Samples the extreme values of the range for each parameter.
        Default is "unc".
    n : int, optional
        Number of samples to generate. Default is 1000. For grids n is the desired number of samples.
    par_to_sample : list of str, optional
        List of parameters to sample. If None, all parameters will be sampled.
    grid_n : int, optional
        Number of samples to generate for each parameter. If None, the number of samples is estimated based on the number of parameters and the desired number of samples.
    

    Returns
    -------
    df_samples : pd.DataFrame
        DataFrame containing the generated samples.


    """

    if isinstance(par_to_sample, str):
        par_to_sample = [par_to_sample]

    if par_to_sample is None:
        par_to_sample = par_space0.keys()

    par_space = {k: v for k, v in par_space0.items() if k in par_to_sample}

    if type not in ["unc", "uniform", "grid", "extremes"]:
        raise ValueError(
            f"Invalid type: {type}. Must be one of 'unc', 'uniform', 'grid', or 'extremes'."
        )

    par_space_ds = {}

    if type == "unc":

        for k, v in par_space.items():
            if "unc_frac" in v and v["unc_frac"] is not None:
                unc_local = v["unc_frac"] * v["mean"]
            else:
                unc_local = v["unc"]

            if "options" in v:
                par_space_ds[k] = np.random.choice(v["range"], p=v["prob"], size=n)
            elif "range" in v:
                if v["unc_type"] == "normal":
                    par_space_ds[k] = np.random.normal(v["mean"], unc_local, size=n)
                elif v["unc_type"] == "uniform":
                    par_space_ds[k] = np.random.uniform(
                        v["range"][0], v["range"][1], size=n
                    )
                elif v["unc_type"] == "exponential":
                    lamda_exp = 1 / v["mean"]
                    par_space_ds[k] = np.random.exponential(lamda_exp, size=n)
                elif v["unc_type"] == "lognormal":
                    mean_log = np.log(
                        v["mean"] ** 2 / np.sqrt(unc_local**2 + v["mean"] ** 2)
                    )
                    sigma_log = np.sqrt(np.log(unc_local**2 / v["mean"] ** 2 + 1))

                    par_space_ds[k] = np.random.lognormal(mean_log, sigma_log, size=n)
            else:
                raise ValueError("par_space needs a range or options.")

    elif type == "uniform":
        for k, v in par_space.items():
            if "options" in v:
                par_space_ds[k] = np.random.choice(v["range"], size=n)
            elif "range" in v:
                par_space_ds[k] = np.random.uniform(
                    v["range"][0], v["range"][1], size=n
                )

    elif type == "grid" or type == "extremes":
        par_space_sets = {}
        if grid_n is None:
            # Estimate grid points for each parameter
            option_ns = [len(v["range"]) for v in par_space.values() if "options" in v]
            grid_ns = [v["grid_n"] for v in par_space.values() if "grid_n" in v]
            
            grid_range_0 = [1 for v in par_space.values() if len(v["range"]) == 1 and 'options' not in v]

            n_dimensions = len(par_space)
            n_dim_left = (
                n_dimensions - len(option_ns) - len(grid_ns) - len(grid_range_0)
            )
            # no range dims
            grid_n = max(
                2,
                round(
                    (n / (np.prod(option_ns) * np.prod(grid_ns))) ** (1 / n_dim_left)
                ),
            )
            grid_n = int(grid_n)

        for k, v in par_space.items():
            if "options" in v:
                par_space_sets[k] = v["range"]
            else:
                if type == "extremes":
                    par_space_sets[k] = np.unique(np.append(v["range"], v["mean"]))
                else:
                    if "grid_n" in v:
                        grid_n_k = v["grid_n"]
                    if v["range"][0] == v["range"][1]:
                        grid_n_k = 1
                    else:
                        grid_n_k = grid_n

                    par_space_sets[k] = np.linspace(
                        v["range"][0], v["range"][1], grid_n_k
                    )

        par_space_ds = generate_combos(par_space_sets, type="")

    # add the other parameters that were not sampled
    for k, v in par_space0.items():
        if k not in par_to_sample:
            par_space_ds[k] = v["mean"]

    # if grid, add ref case at the first row
    if type == "grid":
        ref = {k: v["mean"] for k, v in par_space0.items()}
        ref = pd.DataFrame.from_dict(ref, orient="index").T

        par_space_ds = pd.concat([ref, par_space_ds], ignore_index=True).reset_index(
            drop=True
        )

    # Convert the dictionary of samples to a DataFrame
    df_samples = pd.DataFrame.from_dict(par_space_ds)

    return df_samples


def run_analysis(
        model: object,
        input_stack: object,
        n_samples: object = 2000,
        analyses: object = None,
        par_sensitivity: object = None,
        par_sensitivity_range: object = None,
        par_grid_xy: object = None,
        par_output: object = "y0",
        par_opt: object = "y0",
        data_folder: object = "analysis",
        plotting: object = False,
        save_results: object = False,
        x_range: object = None,
        y_range: object = None,
        parallel : bool = False,
        num_cpus: object = None,
        batch_size: object = None,
) -> object:
    """
    Run various analyses on the model based on the input stack.

    Parameters
    ----------
    model : function
        The model function to be analyzed.
    input_stack : dict
        Dictionary specifying the input parameters and their properties.
    n_samples : int, optional
        Number of samples to generate for analyses. Default is 2000.
    analyses : list of str, optional
        List of analyses to perform. If None, all analyses will be skipped.
        Possible values:
            "estimate": Runs the model with the mean values of the input parameters.
            "estimate_unc": Runs the model with sampled input parameters based on their uncertainty distributions.
            "estimate_unc_extreme_combos": Runs the model with combinations of extreme values of the input parameters.
            "sensitivity_analysis_unc": Performs sensitivity analysis by varying each parameter individually based on its uncertainty distribution.
            "sensitivity_analysis_range": Performs sensitivity analysis by varying each parameter individually over its entire range.
            "sensitivity_analysis_2D": Performs 2D sensitivity analysis by varying two parameters simultaneously over a grid.
            "regular_grid": Runs the model over a regular grid of input parameter values.
            "random_uniform_grid": Runs the model over a grid of randomly sampled input parameter values.

    par_sensitivity : list of str, optional
        List of parameters to perform sensitivity analysis on. If None, no sensitivity analysis will be performed.
    par_sensitivity_range : list of str, optional
        List of tupples with min and max for each parameter to perform sensitivity analysis on. can be none.
    par_grid_xy : list of str, optional
        List of parameters to perform 2D grid analysis on. If None, no 2D grid analysis will be performed.
    par_output : str, optional
        Output variable to analyze. Default is "y0".
    par_opt : str, optional
        Key to use for optimization.
    data_folder : str, optional
        Folder to save analysis outputs. Default is "analysis".
    plotting : bool, optional
        Whether to generate plots for the analyses. Default is False.

    Returns
    -------
    if only a single analysis is run, returns the outputs and output stats.
    otherwise returns None.
    """

    fixed_inputs = {
        k: v["mean"] for k, v in input_stack.items() if len(v["range"]) == 1
    }
    variable_inputs = {k: v for k, v in input_stack.items() if k not in fixed_inputs}

    if analyses is None:
        analyses = []
    elif isinstance(analyses, str):
        analyses = [analyses]
    else:
        # Check that all provided analyses are valid
        valid_analyses = [
            "estimate",
            "estimate_unc",
            "estimate_unc_extreme_combos",
            "sensitivity_analysis_unc",
            "sensitivity_analysis_range",
            "sensitivity_analysis_2D",
            "regular_grid",
            "random_uniform_grid",
        ]
        for analysis in analyses:
            if analysis not in valid_analyses:
                raise ValueError(f"Unknown analysis type: {analysis}")
    if type(par_output) is str:
        par_output = [par_output]

    # straight estimate.
    cases = [{k: v["mean"] for k, v in input_stack.items()}]
    res_0 = run_cases(cases, model)
    if save_results:
        create_dir(os.path.join(data_folder, "estimate"))
        res_0["out"].to_csv(
            os.path.join(data_folder, "estimate", "outputs.csv"), index=False
        )

    if len(analyses) == 1 and analyses[0] == "estimate":
        return res_0

    if "estimate_unc" in analyses:
        cases = generate_samples(input_stack, n=n_samples, type="unc")
        res = run_cases(cases, model, output_stats=True, parallel=parallel, num_cpus=num_cpus, batch_size=batch_size)

        if save_results:
            create_dir(os.path.join(data_folder, "estimate_unc"))
            res["out"].to_csv(
                os.path.join(data_folder, "estimate_unc", "outputs.csv"),
                index=False,
            )
            res["out_stats"].to_csv(
                os.path.join(data_folder, "estimate_unc", "output_stats.csv"),
                index=False,
            )
        if plotting:
            create_dir_safe(os.path.join(data_folder, "estimate_unc"))

            basic_plot_set(
                df=res["out"],
                par=[],
                parz_list=par_output,
                data_folder=os.path.join(data_folder, "estimate_unc"),
                df0=res_0["out"],
            )

    if "estimate_unc_extreme_combos" in analyses:
        cases = generate_samples(input_stack, n=n_samples, type="extremes")
        res = run_cases(cases, model, output_stats=True, parallel=parallel, num_cpus=num_cpus, batch_size=batch_size)
        if save_results:
            create_dir(os.path.join(data_folder, "estimate_unc_extreme_combos"))

            res["out"].to_csv(
                os.path.join(data_folder, "estimate_unc_extreme_combos", "outputs.csv"),
                index=False,
            )
            res["out_stats"].to_csv(
                os.path.join(data_folder, "estimate_unc_extreme_combos", "output_stats.csv"),
                index=False,
            )

    if "sensitivity_analysis_unc" in analyses:
        for par_i in par_sensitivity:
            cases = generate_samples(
                input_stack, n=n_samples, type="unc", par_to_sample=par_i
            )
            res = run_cases(cases, model, output_stats=True, parallel=parallel, num_cpus=num_cpus, batch_size=batch_size)
            if save_results:
                d_ifolder = os.path.join(data_folder, f"sensitivity_analysis_unc_{clean_fld_name(par_i)}")
                create_dir(d_ifolder)
                res["out"].to_csv(
                    os.path.join(d_ifolder, "outputs.csv"),
                    index=False,
                )
                res["out_stats"].to_csv(
                    os.path.join(d_ifolder, "output_stats.csv"),
                    index=False,
                )
            if plotting:

                basic_plot_set(
                    df=res["out"],
                    par=[par_i],
                    parz_list=par_output,
                    data_folder=d_ifolder,
                    df0=res_0["out"],
                )

    if "sensitivity_analysis_range" in analyses:
        for i, par_i in enumerate(par_sensitivity):
            if par_sensitivity_range is not None:
                # hot swaps the range with the min max values
                input_stack[par_i]["range"][0] = par_sensitivity_range[i][0]
                input_stack[par_i]["range"][1] = par_sensitivity_range[i][1]
                
            cases = generate_samples(
                input_stack, n=n_samples, type="grid", par_to_sample=par_i
            )
            res = run_cases(cases, model, output_stats=True, parallel=parallel, num_cpus=num_cpus, batch_size=batch_size)
            d_ifolder = os.path.join(data_folder, f"sensitivity_analysis_range_{clean_fld_name(par_i)}")
            if save_results:
                create_dir(d_ifolder)
                res["out"].to_csv(
                    os.path.join(d_ifolder, "outputs.csv"),
                    index=False,
                )
                res["out_stats"].to_csv(
                    os.path.join(
                        d_ifolder,
                        "output_stats.csv",
                    ),
                    index=False,
                )

            if plotting:
                basic_plot_set(
                    df=res["out"],
                    par=[par_i],
                    parz_list=par_output,
                    data_folder=d_ifolder,
                    df0=res_0["out"],
                )

    if "sensitivity_analysis_2D" in analyses:
        cases = generate_samples(
            input_stack, n=n_samples, type="grid", par_to_sample=par_grid_xy
        )

        res = run_cases(cases, model, output_stats=True, parallel=parallel, num_cpus=num_cpus, batch_size=batch_size)
        create_dir(os.path.join(data_folder, "sensitivity_analysis_2D"))
        res["out"].to_csv(
            os.path.join(data_folder, "sensitivity_analysis_2D", "outputs.csv"),
            index=False,
        )
        res["out_stats"].to_csv(
            os.path.join(data_folder, "sensitivity_analysis_2D", "output_stats.csv"),
            index=False,
        )

        if plotting:
            basic_plot_set(
                df=res["out"],
                par=list(par_grid_xy),
                parz_list=par_output,
                data_folder=os.path.join(data_folder, "sensitivity_analysis_2D"),
            )

    if "regular_grid" in analyses:
        cases = generate_samples(input_stack, n=n_samples, type="grid")
        res = run_cases(cases, model, parallel=parallel, num_cpus=num_cpus, batch_size=batch_size)
        create_dir(os.path.join(data_folder, "regular_grid"))
        res["out"].to_csv(
            os.path.join(data_folder, "regular_grid", "outputs.csv"), index=False
        )
        if plotting:
            basic_plot_set(
                df=res["out"],
                par=list(input_stack.keys()),
                parz_list=par_output,
                data_folder=os.path.join(data_folder, "regular_grid"),
            )

    if "random_uniform_grid" in analyses:
        cases = generate_samples(input_stack, n=n_samples, type="uniform")
        res = run_cases(cases, model, parallel=parallel, num_cpus=num_cpus, batch_size=batch_size)
        create_dir(os.path.join(data_folder, "random_uniform_grid"))
        res["out"].to_csv(
            os.path.join(data_folder, "random_uniform_grid", "outputs.csv"), index=False
        )
        if plotting:
            basic_plot_set(
                df=res["out"],
                par=list(input_stack.keys()),
                parz_list=par_output,
                data_folder=os.path.join(data_folder, "random_uniform_grid"),
            )

    if "GA" in analyses:
        print("Coming soon")

    if "population_rankings" in analyses:
        print("Coming soon")

    if "sobol_indices" in analyses:
        dists = []
        for key, value in input_stack.items():
            if key in variable_inputs:
                if value["unc_type"] == "uniform":
                    dists.append(
                        uniform(
                            loc=value["range"][0],
                            scale=value["range"][1] - value["range"][0],
                        )
                    )
                elif value["unc_type"] == "normal":
                    dists.append(norm(loc=value["mean"], scale=value["unc"]))
                elif value["unc_type"] == "lognormal":
                    mu_log = np.log(
                        value["mean"] ** 2
                        / np.sqrt(value["unc"] ** 2 + value["mean"] ** 2)
                    )
                    sigma_log = np.sqrt(
                        np.log(value["unc"] ** 2 / value["mean"] ** 2 + 1)
                    )
                    dists.append(lognorm(s=sigma_log, scale=np.exp(mu_log)))
                elif value["unc_type"] == "choice":
                    dists.append(norm(loc=value["mean"], scale=value["unc"]))

        rng = np.random.default_rng()

        indices = sobol_indices(func=NEORL_model, n=1024, dists=dists, random_state=rng)
        boot = indices.bootstrap()

        print(indices)

    if len(analyses) == 1:

        res["out_no_unc"] = res_0["out"]
        return res

    return res_0


def create_model_wrap(model,input_stack, value_key, n_samples=100, lamda_w=1, analysis="estimate_unc"):
    """
    Wrap a model to find uncertainty, which is added to the outputs for all the variables.
    """

    # analysis needs to be either estimate_unc, estimate_unc_extreme_combos
    if analysis not in ["estimate_unc", "estimate_unc_extreme_combos"]:
        raise ValueError("analysis must be either estimate_unc or estimate_unc_extreme_combos")
    

    def model_w_unc(x):
        """ 
        the output of the the function is like model output, but with additional values that can be used.
        """
        # replace the values in the input stack with the values in x
 
        for k, v in x.items():
            input_stack[k]["mean"] = v
        
        res = run_analysis(
            model,
            input_stack,
            n_samples=n_samples,
            analyses=[analysis],
            par_output=value_key,

        )

        res_stats = res["out_stats"]
        res_stats["lambda"] = res_stats["mean"] - lamda_w * res_stats["std"] ** 2
        res_stats["sharpe"] = res_stats["mean"] / res_stats["std"]

        model_out = {}
        model_out = df_to_dict(res_stats, model_out)
   
        return model_out

    return model_w_unc


def prep_model_for_NEORL(model, input_stack, value_key):
    # split input stack into fixed and variable, based on if unc is 0 or if range is length 1
    fixed_inputs = {
        k: v["mean"] for k, v in input_stack.items() if len(v["range"]) == 1
    }
    variable_inputs = {k: v for k, v in input_stack.items() if k not in fixed_inputs}

    NEORL_model = create_NEORL_funwrap(
        model,
        value_key=value_key,
        variable_inputs=variable_inputs.keys(),
        fixed_inputs=fixed_inputs,
    )

    BOUNDS = NEORL_getbounds(variable_inputs)

    return NEORL_model, BOUNDS


def run_NEORL(model, input_stack, value_key):

    NEORL_model, BOUNDS = prep_model_for_NEORL(model, input_stack, value_key)

    es = ES(mode='min', fit=NEORL_model, cxmode='blend', bounds=BOUNDS, 
                 lambda_=60, mu=30, cxpb=0.7, mutpb=0.2,ncores=8, seed=1)
    x_es, y_es, es_hist=es.evolute(ngen=200, verbose=True)

    return x_es, y_es, es_hist

if __name__ == "__main__":

    def model(x):
        out = {}
        out["y0"] = x["x0"] ** 2 + np.exp(x["x1"]) + x["x3"]
        out["y1"] = x["x0"] + x["x1"] + x["x2"] + x["x3"]
        time.sleep(.05)
        return out

    # Dictionary specifying variables with uncertainties
    # mean, unc, unc_range (tolerance or 3 sigma), bounds (minimum and maximum value)
    input_stack = {
        "x0": {
            "mean": 1.0,
            "unc": 0.2,
            "range": [0, 5],
            "bounds": [0, 100],
            "unc_type": "normal",
        },
        "x1": {"mean": 1.0, "unc": 0.2, "range": [0, 3], "unc_type": "normal"},
        "x2": 3.0,
        "x3": 4,
        "x4": "a",
        "x5": {
            "mean": "a",
            "range": ["a", "b"],
            "options": ["a", "b", "c"],
            "prob": [0.1, 0.2, 0.7],
            "unc_type": "choice",
        },
        "x6": {
            "mean": "a",
            "options": ["a", "b", "c"],
            "unc_type": "choice",
        },
        "x7": {
            "mean": "a",
            "unc": [0.2, 0.8],
            "range": ["a", "b"],
            "options": ["a", "b", "c"],
            "unc_type": "choice",
        },
    }

    input_stack = process_input_stack(input_stack)
    print(input_stack)

    if 1 == 0:
        # NEORL model

        # split input stack into fixed and variable, based on if unc is 0 or if range is length 1
        fixed_inputs = {
            k: v["mean"] for k, v in input_stack.items() if len(v["range"]) == 1
        }
        variable_inputs = {
            k: v for k, v in input_stack.items() if k not in fixed_inputs
        }
        par_opt = "y0"
        print("fixed_inputs", fixed_inputs)
        print("variable_inputs", variable_inputs)

        from NEORL_wrap import create_NEORL_funwrap, NEORL_getbounds

        NEORL_model = create_NEORL_funwrap(
            model,
            par_opt=par_opt,
            input_key=variable_inputs.keys(),
            fixed_inputs=fixed_inputs,
        )

        BOUNDS = NEORL_getbounds(variable_inputs)

        # try NEORL model with values within the bounds
        # Generate values within the bounds
        x_values = []
        for key, bound in BOUNDS.items():
            if bound[0] == "float":
                x_values.append(np.random.uniform(bound[1], bound[2]))
            elif bound[0] == "int":
                x_values.append(np.random.randint(bound[1], bound[2]))
            elif bound[0] == "grid":
                x_values.append(np.random.choice(bound[1]))

        print(x_values)

        # Call the NEORL model with the generated values
        result = NEORL_model(x_values)
        print("NEORL model result:", result)

        print("BOUNDS", BOUNDS)

    # pandas print entire columns no abbrreb
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.width', None)

    # run each analysis
    run_analysis(model=model, input_stack=input_stack, analyses=["estimate"],  )
    res = run_analysis(
        model,
        input_stack,
        n_samples=100,
        analyses=["estimate_unc"],
        par_output="y0",
        plotting=True,
        parallel=True,
        batch_size=None,
        # num_cpus=1,
    )

    print(res)
    if 1==0:
        # Test model wraps

        model_w_unc = create_model_wrap(
            model,
            input_stack=input_stack,
            analysis="estimate_unc",
            n_samples=1000,
            lamda_w=1,
            value_key="y0",
        )

        model_w_extremes = create_model_wrap(
            model,
            input_stack=input_stack,
            analysis="estimate_unc_extreme_combos",
            n_samples=100,
            lamda_w=1,
            value_key="y0",
        )

        run_analysis(
            model_w_unc,
            input_stack,
            n_samples=10000,
            analyses=["sensitivity_analysis_2D"],
            par_grid_xy=["x0", "x1"],
            par_output="y0_sharpe",
            plotting=True,
        )

    # run_analysis(model, input_stack, n_samples=1000, analyses=["estimate_unc_extreme_combos"],  par_output="y0")

    # run_analysis(model, input_stack, n_samples=1000, analyses=["sensitivity_analysis_unc"], par_sensitivity=["x0", "x1"], par_grid_xy=["x0", "x1"], par_output="y0")
    # run_analysis(model, input_stack, n_samples=1000, analyses=["sensitivity_analysis_range"], par_sensitivity=["x0", "x1"], par_grid_xy=["x0", "x1"], par_output="y0")

    # run_analysis(model, input_stack, n_samples=1000, analyses=["sensitivity_analysis_2D"],  par_grid_xy=["x0", "x1"], par_output="y0")
    # run_analysis(model, input_stack, n_samples=1000,
    #              analyses=["regular_grid"],  par_output="y0")
    # run_analysis(model, input_stack, n_samples=1000, analyses=["random_uniform_grid"], par_output="y0")

    # run_analysis(model, input_stack, n_samples=1000, analyses=["GA"], par_sensitivity=["x0", "x1"], par_grid_xy=["x0", "x1"], par_output="y0")
    # run_analysis(model, input_stack, n_samples=1000, analyses=["population_rankings"], par_sensitivity=["x0", "x1"], par_grid_xy=["x0", "x1"], par_output="y0")

    # run_analysis(model, input_stack, n_samples=1000, analyses=["sobol_indices"], par_sensitivity=["x0", "x1"], par_grid_xy=["x0", "x1"], par_output="y0")
