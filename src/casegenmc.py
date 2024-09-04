
import numpy as np
import pandas as pd
from util import *
from os.path import join as pjoin
from scipy.stats import sobol_indices, uniform, norm, lognorm
from plotting_base import *
import itertools

PARALLEL = 0

if PARALLEL:
    import ray

    ray.init()

    @ray.remote
    class FileWriter:
        def __init__(self, file_name):
            self.file_name = file_name
            self.header_written = False

        def write(self, row):
            if isinstance(row, dict):
                row = pd.DataFrame([row])
            with open(self.file_name, "a") as f:
                if not self.header_written:
                    row.to_csv(f, index=True)
                    self.header_written = True
                else:
                    # row_vals = row.loc[["v"], :]
                    row.to_csv(f, header=False, index=True)

    @ray.remote
    def run_inputP(i, case, model, writer=None):
        """
        Wroker function for parallel processing. This function is called by the ray library.

        :param i:
        :param input_dict:
        :param data_out_dir:
        :param save_all_dfs:
        :param save_figs:
        :param writer:
        :return:
        """

        i = model(case)
        ray.get(writer.write.remote(i))
        return i

    @timer
    def run_inputS(i, case, model, writer=None):
        i = model(case)
        writer.write.remote(i)
        return i





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
                isinstance(v, (int, float)) and isinstance(
                    base_inputs[k], (int, float))
            )
        ):
            raise TypeError(
                f"Type mismatch for parameter {k}. Expected {type(base_inputs[k])}, got {type(v)}."
            )

    # check there are no repeats
    if len(add_inputs.keys()) != len(set(add_inputs.keys())):
        raise ValueError("There are repeats in the parameter space")

    return


def process_input_stack(input_stack, default_unc_type="normal", default_unc_frac=0):
    for key, value in input_stack.items():
        if isinstance(value, (int, float, str, bool)):
            input_stack[key] = {
                "mean": value,
                "unc": 0,
                "options": [value],

                # "unc_type": "choice" if not isinstance(value, (int, float)) else "normal",
                "type": 'options',
                # "bounds": None if not isinstance(value, (int, float)) else [-np.inf, np.inf]
            }

        value = input_stack[key]

        if isinstance(value, dict):

            if "options" in value:
                value["unc_type"] = "choice"
                value["bounds"] = None
                value["type"] = 'options'
                value["unc_frac"] = None
                value["unc"] = None

                if "range" in value:
                    # check that range is in options
                    if not all(v in value["options"] for v in value["range"]):
                        raise ValueError(
                            f"Range {value['range']} is not in options {value['options']} for key '{key}'")
                else:
                    value["range"] = [value["mean"]]

                if "prob" in value:
                    if not (len(value["options"]) == len(value["prob"])):
                        raise ValueError(
                            f"Length of 'prob' ({len(value['prob'])}) must match length of 'options' ({len(value['options'])}) for key '{key}'")
                else:
                    # uniform distribution
                    value["prob"] = np.ones(
                        len(value["range"]))/len(value["range"])
            else:
                # convert unc_frac to unc or vice versa

                if "unc_frac" in value:
                    value["unc"] = value["unc_frac"] * value["mean"]
                elif "unc" in value:
                    value["unc_frac"] = value["unc"] / value["mean"]
                else:
                    value["unc_frac"] = default_unc_frac
                    value["unc"] = default_unc_frac * value["mean"]

                if "range" not in value:
                    value["range"] = [value["mean"] - 3 *
                                      value["unc"], value["mean"] + 3*value["unc"]]
                else:
                    # check that range is in bounds
                    if "bounds" in value:
                        if not (value["bounds"][0] <= value["range"][0] <= value["bounds"][1]):
                            raise ValueError(
                                f"Range {value['range']} is not in bounds {value['bounds']} for key '{key}'")

                if "unc_type" not in value:
                    if isinstance(value["mean"], (int, float)):
                        value["unc_type"] = default_unc_type

                if 'bounds' not in value:
                    value['bounds'] = [0, 100*value['mean']]
                if 'type' not in value:
                    value['type'] = type(value['mean']).__name__

        # check that mean is in range and bounds
        if input_stack[key]['type'] == 'float' or input_stack[key]['type'] == 'int':
            if not (input_stack[key]["range"][0] <= input_stack[key]["mean"] <= input_stack[key]["range"][1]):
                raise ValueError(
                    f"Mean {input_stack[key]['mean']} is not in range {input_stack[key]['range']} for key '{key}'")
            if "bounds" in input_stack[key] and not (input_stack[key]["bounds"][0] <= input_stack[key]["mean"] <= input_stack[key]["bounds"][1]):
                raise ValueError(
                    f"Mean {input_stack[key]['mean']} is not in bounds {input_stack[key]['bounds']} for key '{key}'")

    return input_stack


def run_cases(inputs, model, output_stats=False, parallel=PARALLEL):
    """

    """
    data_out_dir = "./data"
    create_dir_safe(data_out_dir)
    # time this operation
    start_time = time.time()

    outputs = {}

    if isinstance(inputs, list):
        inputs = pd.DataFrame(inputs)

    if PARALLEL:
        file_name_summary = pjoin(data_out_dir, "summary.csv")
        with open(file_name_summary, "w") as f:
            pass
        writer = FileWriter.remote(file_name_summary)

        if not parallel:  # run in series
            for i, case in inputs.iterrows():
                outputs[i] = run_inputS(i, case, model, writer)

        else:  # run in parallel
            # Launch four parallel square tasks.
            for i, case in inputs.iterrows():
                futures = run_inputP.remote(i, case, model, writer)

            print(ray.get(futures))
    else:
        for i, case in inputs.iterrows():
            outputs[i] = model(case)
    end_time = time.time()
    print("inputs count", len(inputs))
    print("--- %s seconds per run---" %
          ((end_time - start_time) / len(inputs)))
    print("--- %s seconds ---" % (end_time - start_time))

    outputs = pd.DataFrame.from_dict(outputs).T

    # combine with cases
    outputs = pd.concat([inputs, outputs], axis=1)
    output_stats = {}
    
    # if output_stats, create a summary dic calculate mean, std, min, max, and std_fractional for each output
    if output_stats:
        for output in outputs.columns:

            output_stats[output] = {}
            if pd.api.types.is_numeric_dtype(outputs[output]):
                output_stats[output]["mean"] = outputs[output].mean()
                output_stats[output]["std"] = outputs[output].std()
                output_stats[output]["std_frac"] = outputs[output].std(
                ) / output_stats[output]["mean"]
                output_stats[output]["min"] = outputs[output].min()
                output_stats[output]["max"] = outputs[output].max()
                output_stats[output]["frac_minus"] = (
                    output_stats[output]["min"] - output_stats[output]["mean"]) / output_stats[output]["mean"]
                output_stats[output]["frac_plus"] = (
                    output_stats[output]["max"] - output_stats[output]["mean"]) / output_stats[output]["mean"]
            else:
                output_stats[output]["mode"] = outputs[output].mode()[0]

        output_stats = pd.DataFrame.from_dict(output_stats).T

        return outputs, output_stats
    return outputs


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
        par_tests = np.vstack(par_tests)

        combos_data = pd.DataFrame(
            data=par_tests, columns=parameter,).apply(pd.to_numeric, errors='ignore')

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


@timer
def generate_samples(par_space0, type="unc", n=1000, par_to_sample=None,
                     grid_n=None):
    """
    Generates samples from a parameter space.

    Parameters
    ----------
    par_space : dict
        Dictionary specifying the parameter space.
    type : str, optional
        Type of sampling to perform. Must be one of:
        - "unc": Samples based on the uncertainty type and range specified for each parameter in par_space. 
                 For parameters with "unc_type" specified, it will sample from the corresponding distribution 
                 (e.g., normal with mean and standard deviation, uniform over range, etc.).
                 For parameters with "options" specified, it will randomly select from the options according to their probabilities.
        - "uniform": Samples uniformly over the range specified for each parameter, ignoring any uncertainty type. 
        - "grid": Generates a regular grid of samples over the range of each parameter.
        - "extremes": Samples the extreme values of the range for each parameter.
        Default is "unc".
    n : int, optional
        Number of samples to generate. Default is 1000. For grids n is the desired number of samples.

    Returns
    -------
    par_space_ds : dict
        Dictionary containing the generated samples for each parameter.


    Generates samples from a parameter space.

    Either
    - use the unc and unc_type
    - force uniform sampling
    - generate grid points


    generate_samples(par_space, n=1000)

    """

    if par_to_sample is None:
        par_to_sample = par_space0.keys()

    par_space = {k: v for k, v in par_space0.items() if k in par_to_sample}

    if type not in ["unc", "uniform", "grid", "extremes"]:
        raise ValueError(
            f"Invalid type: {type}. Must be one of 'unc', 'uniform', 'grid', or 'extremes'.")

    par_space_ds = {}

    if type == "unc":

        for k, v in par_space.items():

            if "options" in v:
                par_space_ds[k] = np.random.choice(
                    v["range"], p=v["prob"], size=n)
            elif "range" in v:
                if v["unc_type"] == "normal":
                    par_space_ds[k] = np.random.normal(
                        v["mean"], v["unc"], size=n)
                elif v["unc_type"] == "uniform":
                    par_space_ds[k] = np.random.uniform(
                        v["range"][0], v["range"][1], size=n)
                elif v["unc_type"] == "exponential":
                    lamda_exp = 1/v["mean"]
                    par_space_ds[k] = np.random.exponential(lamda_exp, size=n)
                elif v["unc_type"] == "lognormal":
                    mean_log = np.log(
                        v["mean"]**2 / np.sqrt(v["unc"]**2 + v["mean"]**2))
                    sigma_log = np.sqrt(np.log(v["unc"]**2 / v["mean"]**2 + 1))

                    par_space_ds[k] = np.random.lognormal(
                        mean_log, sigma_log, size=n)
            else:
                raise ValueError("par_space needs a range or options.")

    elif type == "uniform":
        for k, v in par_space.items():
            if "options" in v:
                par_space_ds[k] = np.random.choice(
                    v["range"],  size=n)
            elif "range" in v:
                par_space_ds[k] = np.random.uniform(
                    v["range"][0], v["range"][1], size=n)

    elif type == "grid" or type == "extremes":
        par_space_sets = {}

        if grid_n is None:
            # Estimate grid points for each parameter
            option_ns = [len(v["range"])
                         for v in par_space.values() if "options" in v]
            grid_ns = [v["grid_n"]
                       for v in par_space.values() if "grid_n" in v]
            grid_range_0 = [
                1 for v in par_space.values() if len(v["range"]) == 1]

            n_dimensions = len(par_space)
            n_dim_left = n_dimensions - \
                len(option_ns) - len(grid_ns) - len(grid_range_0)

            # no range dims

            grid_n = max(
                2, round((n / (np.prod(option_ns) * np.prod(grid_ns))) ** (1 / n_dim_left)))
            grid_n = int(grid_n)

        for k, v in par_space.items():
            if "options" in v:
                par_space_sets[k] = v["range"]
            else:
                if type == "extremes":
                    par_space_sets[k] = np.unique(
                        np.append(v["range"], v["mean"]))
                else:
                    if "grid_n" in v:
                        grid_n_k = v["grid_n"]
                    if v["range"][0] == v["range"][1]:
                        grid_n_k = 1
                    else:
                        grid_n_k = grid_n

                    par_space_sets[k] = np.linspace(
                        v["range"][0], v["range"][1], grid_n_k)

        par_space_ds = generate_combos(par_space_sets, type="")

    # add the other parameters that were not sampled
    for k, v in par_space0.items():
        if k not in par_to_sample:
            par_space_ds[k] = v["mean"]

    # if grid, add ref case at the first row
    if type == "grid":
        ref = {k: v["mean"] for k, v in par_space0.items()}
        ref = pd.DataFrame.from_dict(ref, orient="index").T

        par_space_ds = pd.concat(
            [ref, par_space_ds], ignore_index=True).reset_index(drop=True)

    # Convert the dictionary of samples to a DataFrame
    df_samples = pd.DataFrame.from_dict(par_space_ds)

    return df_samples


def run_analysis(model, input_stack, n_samples=2000, analyses=None, par_sensitivity=None, par_grid_xy=None, par_output="y0", par_opt="y0", data_folder="data"):
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
            "single_estimate": Runs the model with the mean values of the input parameters.
            "estimate_with_unc": Runs the model with sampled input parameters based on their uncertainty distributions.
            "estimate_with_unc_combos": Runs the model with combinations of extreme values of the input parameters.
            "sensitivity_analysis_unc": Performs sensitivity analysis by varying each parameter individually based on its uncertainty distribution.
            "sensitivity_analysis_range": Performs sensitivity analysis by varying each parameter individually over its entire range.
            "sensitivity_analysis_2D": Performs 2D sensitivity analysis by varying two parameters simultaneously over a grid.
            "regular_grid": Runs the model over a regular grid of input parameter values.
            "random_uniform_grid": Runs the model over a grid of randomly sampled input parameter values.
            "GA": Performs optimization using a genetic algorithm over the bounds of the input parameters.
            "population_rankings": Analyzes the model output for different subpopulations of the input space.
            "sobol_indices": Computes Sobol sensitivity indices for the input parameters.
    par_sensitivity : list of str, optional
        List of parameters to perform sensitivity analysis on. If None, no sensitivity analysis will be performed.
    par_grid_xy : list of str, optional
        List of parameters to perform 2D grid analysis on. If None, no 2D grid analysis will be performed.
    par_output : str, optional
        Output variable to analyze. Default is "y0".
    par_opt : str, optional
        Key to use optimization.
    data_folder : str, optional
        Folder to save analysis outputs. Default is "data".

    Returns
    -------
    None
    """

    fixed_inputs = {k: v["mean"] for k, v in input_stack.items()
                    if len(v["range"]) == 1}
    variable_inputs = {k: v for k,
                       v in input_stack.items() if k not in fixed_inputs}

    if analyses is None:
        analyses = []

    if "estimate" in analyses:
        cases = [{k: v["mean"] for k, v in input_stack.items()}]
        outputs = run_cases(cases, model)
        create_dir(os.path.join(data_folder, "estimate"))
        outputs.to_csv(os.path.join(data_folder, "estimate", "outputs.csv"), index=False)

    if "estimate_with_unc" in analyses:
        cases = generate_samples(input_stack, n=n_samples, type="unc")
        outputs, output_stats = run_cases(cases, model, output_stats=True)
        create_dir(os.path.join(data_folder, "estimate_with_unc"))
        outputs.to_csv(os.path.join(data_folder, "estimate_with_unc", "outputs.csv"), index=False)
        output_stats.to_csv(os.path.join(data_folder, "estimate_with_unc", "output_stats.csv"), index=False)
        basic_plot_set(df=outputs,
                       par=list(variable_inputs.keys()),
                       parz=par_output, data_folder=os.path.join(data_folder, "estimate_with_unc"))

    if "estimate_with_unc_combos" in analyses:
        cases = generate_samples(input_stack, n=n_samples, type="extremes")
        outputs, output_stats = run_cases(cases, model, output_stats=True)
        create_dir(os.path.join(data_folder, "estimate_with_unc_combos"))
        outputs.to_csv(os.path.join(data_folder, "estimate_with_unc_combos", "outputs.csv"), index=False)
        output_stats.to_csv(os.path.join(data_folder, "estimate_with_unc_combos", "output_stats.csv"), index=False)

    if "sensitivity_analysis_unc" in analyses:
        for par_i in par_sensitivity:
            cases = generate_samples(
                input_stack, n=n_samples, type="unc", par_to_sample=par_i)
            outputs, output_stats = run_cases(
                cases, model, output_stats=True)
            create_dir(os.path.join(data_folder, f"sensitivity_analysis_unc_{par_i}"))
            outputs.to_csv(os.path.join(data_folder, f"sensitivity_analysis_unc_{par_i}", "outputs.csv"), index=False)
            output_stats.to_csv(os.path.join(data_folder, f"sensitivity_analysis_unc_{par_i}", "output_stats.csv"), index=False)

    if "sensitivity_analysis_range" in analyses:
        for par_i in par_sensitivity:
            cases = generate_samples(
                input_stack, n=n_samples, type="grid", par_to_sample=par_i)
            outputs, output_stats = run_cases(
                cases, model, output_stats=True)
            create_dir(os.path.join(data_folder, f"sensitivity_analysis_range_{par_i}"))
            outputs.to_csv(os.path.join(data_folder, f"sensitivity_analysis_range_{par_i}", "outputs.csv"), index=False)
            output_stats.to_csv(os.path.join(data_folder, f"sensitivity_analysis_range_{par_i}", "output_stats.csv"), index=False)
            print(output_stats)

    if "sensitivity_analysis_2D" in analyses:
        cases = generate_samples(
            input_stack, n=n_samples, type="grid", par_to_sample=par_grid_xy)

        outputs, output_stats = run_cases(cases, model, output_stats=True)
        create_dir(os.path.join(data_folder, "sensitivity_analysis_2D"))
        outputs.to_csv(os.path.join(data_folder, "sensitivity_analysis_2D", "outputs.csv"), index=False)
        output_stats.to_csv(os.path.join(data_folder, "sensitivity_analysis_2D", "output_stats.csv"), index=False)
        basic_plot_set(df=outputs,
                       par=list(par_grid_xy),
                       parz=par_output, data_folder=os.path.join(data_folder, "sensitivity_analysis_2D"))

    if "regular_grid" in analyses:
        cases = generate_samples(input_stack, n=n_samples, type="grid")
        outputs = run_cases(cases, model)
        create_dir(os.path.join(data_folder, "regular_grid"))
        outputs.to_csv(os.path.join(data_folder, "regular_grid", "outputs.csv"), index=False)
        basic_plot_set(df=outputs, par=list(input_stack.keys()), parz=par_output, data_folder=os.path.join(data_folder, "regular_grid"))

    if "random_uniform_grid" in analyses:
        cases = generate_samples(input_stack, n=n_samples, type="uniform")
        outputs = run_cases(cases, model)
        create_dir(os.path.join(data_folder, "random_uniform_grid"))
        outputs.to_csv(os.path.join(data_folder, "random_uniform_grid", "outputs.csv"), index=False)
        basic_plot_set(df=outputs, par=list(input_stack.keys()), parz=par_output, data_folder=os.path.join(data_folder, "random_uniform_grid"))


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
                        uniform(loc=value["range"][0], scale=value["range"][1]-value["range"][0]))
                elif value["unc_type"] == "normal":
                    dists.append(
                        norm(loc=value["mean"], scale=value["unc"]))
                elif value["unc_type"] == "lognormal":
                    mu_log = np.log(value["mean"]**2 / np.sqrt(value["unc"]**2 + value["mean"]**2))
                    sigma_log = np.sqrt(np.log(value["unc"]**2 / value["mean"]**2 + 1))
                    dists.append(
                        lognorm(s=sigma_log, scale=np.exp(mu_log)))
                elif value["unc_type"] == "choice":
                    dists.append(
                        norm(loc=value["mean"], scale=value["unc"]))

        rng = np.random.default_rng()

        indices = sobol_indices(
            func=NEORL_model, n=1024,
            dists=dists,
            random_state=rng
        )
        boot = indices.bootstrap()

        print(indices)


    # return outputs and output_stats if available in dict
    return
    
    

if __name__ == "__main__":

    def model(x):
        out = {}
        out["y0"] = x["x0"]**2 + np.exp(x["x1"]) + x['x3']
        out["y1"] = x["x0"] + x["x1"] + x["x2"] + x["x3"]
        return out

    input_stack = {"x0": 1, "x1": 2, "x2": 3., "x3": 4, 'x4': 'a'}

    # unc is 1 sigma for unc, and used for unc_type = normal
    # range is for evals, defaults to 3 sigma, and is used for uniform
    # bound is for optimization

    # Dictionary specifying variables with uncertainties
    # mean, unc, unc_range (tolerance or 3 sigma), bounds (minimum and maximum value)
    input_stack = {
        "x0": {"mean": 1., "unc": .2, 'range': [0, 5], 'bounds': [0, 100], 'unc_type': 'normal'},
        "x1": {"mean": 1., "unc": .2, 'range': [0, 3], 'unc_type': 'normal'},
        "x2": 3., "x3": 4, 'x4': 'a',
        "x5": {"mean": "a",  'range': ["a", "b"], "options": ["a", "b", "c"], "unc_type": "choice", },
        "x6": {"mean": "a",   "options": ["a", "b", "c"], "unc_type": "choice", },

    }

    input_stack = process_input_stack(input_stack)
    print('aaa')
    print(input_stack)

    if 1 == 0:
        # NEORL model

        # split input stack into fixed and variable, based on if unc is 0 or if range is length 1
        fixed_inputs = {k: v["mean"] for k, v in input_stack.items()
                        if len(v["range"]) == 1}
        variable_inputs = {k: v for k,
                           v in input_stack.items() if k not in fixed_inputs}
        par_opt = "y0"
        print("fixed_inputs", fixed_inputs)
        print("variable_inputs", variable_inputs)

        from NEORL_wrap import create_neorl_function_dictIO, NEORL_getbounds
        NEORL_model = create_neorl_function_dictIO(
            model, par_opt=par_opt, input_key=variable_inputs.keys(), fixed_inputs=fixed_inputs)

        BOUNDS = NEORL_getbounds(variable_inputs)

        # try NEORL model with values within the bounds
        # Generate values within the bounds
        x_values = []
        for key, bound in BOUNDS.items():
            if bound[0] == 'float':
                x_values.append(np.random.uniform(bound[1], bound[2]))
            elif bound[0] == 'int':
                x_values.append(np.random.randint(bound[1], bound[2]))
            elif bound[0] == 'grid':
                x_values.append(np.random.choice(bound[1]))

        print(x_values)

        # Call the NEORL model with the generated values
        result = NEORL_model(x_values)
        print("NEORL model result:", result)

        print("BOUNDS", BOUNDS)

    # run each analysis
    # run_analysis(model=model, input_stack=input_stack, n_samples=1000, analyses=["estimate"],  par_output="y0")
    # run_analysis(model, input_stack, n_samples=1000, analyses=["estimate_with_unc"], par_sensitivity=["x0", "x1"], par_grid_xy=["x0", "x1"], par_output="y0")
    # run_analysis(model, input_stack, n_samples=1000, analyses=["estimate_with_unc_combos"], par_sensitivity=["x0", "x1"], par_grid_xy=["x0", "x1"], par_output="y0")

    # run_analysis(model, input_stack, n_samples=1000, analyses=["sensitivity_analysis_unc"], par_sensitivity=["x0", "x1"], par_grid_xy=["x0", "x1"], par_output="y0")
    # run_analysis(model, input_stack, n_samples=1000, analyses=["sensitivity_analysis_range"], par_sensitivity=["x0", "x1"], par_grid_xy=["x0", "x1"], par_output="y0")

    # run_analysis(model, input_stack, n_samples=1000, analyses=["sensitivity_analysis_2D"],  par_grid_xy=["x0", "x1"], par_output="y0")
    # run_analysis(model, input_stack, n_samples=1000,
    #              analyses=["regular_grid"],  par_output="y0")
    run_analysis(model, input_stack, n_samples=1000, analyses=["random_uniform_grid"], par_output="y0")

    # run_analysis(model, input_stack, n_samples=1000, analyses=["GA"], par_sensitivity=["x0", "x1"], par_grid_xy=["x0", "x1"], par_output="y0")
    # run_analysis(model, input_stack, n_samples=1000, analyses=["population_rankings"], par_sensitivity=["x0", "x1"], par_grid_xy=["x0", "x1"], par_output="y0")

    # run_analysis(model, input_stack, n_samples=1000, analyses=["sobol_indices"], par_sensitivity=["x0", "x1"], par_grid_xy=["x0", "x1"], par_output="y0")
