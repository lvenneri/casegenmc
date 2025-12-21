# CASEGEN MC

Probe a model to see the possibilities. Takes model and input dictionary and evaluates cases to explore the model space
using grids, random sampling, and optimization techniques.

The input dictionary is defined by the user with mean value, uncertainty, uncertainty distribution, range, and bounds.
Sparse definition okay and assumes 0 unc and range by default. Numerical and categorical parameters are supported. For
categorical parameters, the range is defined as a list of options (subset of options), and the uncertainty distribution
is defined as "choice" with unc defining the probability of each option.

Includes matplotlib utility functions for standard plotting using toggle "plotting" which calls the function
basic_plot_set().

## Defining model inputs:

mean: float or array_like
Mean value of the parameter.

unc: float or array_like, optional
Standard deviation of the parameter. Only used if "unc_frac" is not defined.

unc_frac: float or array_like, optional
Fraction of the mean to use as the standard deviation. Only used if "unc" is not defined.

range: float or array_like, optional
Range of the parameter used for regular grid and random uniform grid. Will default to the mean +/- 3x the unc if not
defined.

bounds: float or array_like, optional
Bounds of the parameter. Used for optimization. Will default to [0, 100x the mean] if not defined.

unc_type: str, optional
Type of uncertainty distribution. If not defined, it is assumed to be uniform. Options: normal, lognormal, choice, exponential. Can add more, but working with normals is convenient for now.


## Analysis Types:

| Analysis Type                 | Description                                                                                                 |
|-------------------------------|-------------------------------------------------------------------------------------------------------------|
| `estimate`                    | Runs the model with the mean values of the input parameters.                                                |
| `estimate_unc`                | Runs the model with sampled input parameters based on their uncertainty distributions.                      |
| `estimate_unc_extreme_combos` | Runs the model with combinations of extreme values of the input parameters.                                 |
| `sensitivity_analysis_unc`    | Performs sensitivity analysis by varying each parameter individually based on its uncertainty distribution. |
| `sensitivity_analysis_range`  | Performs sensitivity analysis by varying each parameter individually over its entire range.                 |
| `sensitivity_analysis_2D`     | Performs 2D sensitivity analysis by varying two parameters simultaneously over a grid.                      |
| `regular_grid`                | Runs the model over a regular grid of input parameter values.                                               |
| `random_uniform_grid`         | Runs the model over a grid of randomly sampled input                                                        

## Install

```
pip install casegenmc
```

## Use

```python
import casegenmc as cgm
import numpy as np

cgm.init_casegenmc(setup_tex=False, fontsize=8, figsize=[6, 6])
print("\n--- 1. Evaluating a model with uncertainty ---")
# Define a model with uncertainties on the inputs
def model(x):
    # multiple input type model and input_stack example
    out = {}
    out["y0"] = x["x0"] ** 2 + np.exp(x["x1"]) + x['x3']
    out["y1"] = x["x0"] + x["x1"] + x["x2"] + x["x3"]
    return out

input_stack = {
    "x0": {"mean": 1., "unc": .2, 'range': [0, 5], 'bounds': [0, 100], 'unc_type': 'normal'},
    "x1": {"mean": 1., "unc": .2, 'range': [0, 3], 'unc_type': 'lognormal'},
    "x2": 3.,
    "x3": 4,
    "x4": {"mean": "a", 'range': ["a", "b"], "options": ["a", "b", "c"], "unc_type": "choice", },
    "x5": {"mean": "a", 'unc': [.2, .8], 'range': ["a", "b"], "options": ["a", "b", "c"], "unc_type": "choice", },
}

# Pre-process input stack
input_stack = cgm.process_input_stack(input_stack)
print(input_stack)

# Evaluate the model with the input_stack.
cgm.run_analysis(model=model, input_stack=input_stack, analyses=["estimate"])

# Estimate with uncertainty.
cgm.run_analysis(model, input_stack, n_samples=1000, analyses=["estimate_unc"], par_output="y0", plotting=True,
                 save_results=True)

# Estimate with uncertainty combinations.
cgm.run_analysis(model, input_stack, n_samples=1000, analyses=["estimate_unc_extreme_combos"], par_output="y0")

# 2d sensitivity analysis and analysis w.r.t 1 output parameter.
cgm.run_analysis(
    model,
    input_stack,
    n_samples=1000,
    analyses=["sensitivity_analysis_2D"],
    par_grid_xy=["x0", "x1"],
    par_output="y0",
    plotting=True,
    parallel=False,
    num_cpus=None,  # all
    batch_size=None,  # all
)
# sample a regular grid defined by range.
cgm.run_analysis(model, input_stack, n_samples=1000, analyses=["regular_grid"], par_output="y0")


```

## Parallel case evals
run_analysis can run cases in parallel using Ray. To enable parallel processing, set `parallel=True` and specify the number of CPUs to use with `num_cpus`. If `num_cpus` is set to `None`, all available CPUs will be used. You can also specify the `batch_size` for distributing tasks among workers. If `batch_size` is set to `None`, all are batchted together.


## Optimization wrappers
run_analysis can be performed using scipy or NEORL (separate install) or scipy optimizers. Wrapper classes are provided to interface with these optimizers. The wrapper class sets includes a mode for minimization or maximization, and rectifies all models to minimization problems.

```python
print("\n--- 2. Optimization Examples (Ackley Function) ---")
def model_Ackley(param_dict, **kwargs):
    """
    Ackley Function. Global Minimum is 0 at x = [0, 0, ..., 0].
    """
    relevant_keys = sorted([k for k in param_dict if k.startswith('x')])
    x = np.array([param_dict[k] for k in relevant_keys])

    d = len(x)
    term1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / d))
    term2 = -np.exp(np.sum(np.cos(2 * np.pi * x)) / d)
    y = term1 + term2 + 20 + np.e

    # Standard Ackley is a MINIMIZATION problem.
    # We return positive 'y' and set wrappers to 'minimize'.
    return {"y0": y}

input_stack_ackley = {
    f"x{i}": {"mean": 5., "range": [-32.768, 32.768]} for i in range(5)
}

# define what to optimize, which parameters to vary, and which are fixed
par_opt = 'y0'  # output parameter to optimize, value function
vary_keys = ['x0', 'x1', 'x2', 'x3', 'x4']
variable_inputs = {name: input_stack_ackley[name] for name in input_stack_ackley}
fixed_inputs = {k: v['mean'] for k, v in input_stack_ackley.items() if k not in vary_keys}

if 1==0:
    print("\n[NEORL] Running Optimization... Requires separate installation.")
    import neorl
    # split input stack into fixed and variable, based on vary_keys
    NEORL_model = cgm.NeorlWrapper(model_Ackley, value_key=par_opt, variable_inputs=variable_inputs.keys(),
                                   fixed_inputs=fixed_inputs, mode='minimize')

    BOUNDS = cgm.NEORL_getbounds(variable_inputs)
    es = neorl.ES(mode='min', fit=neorl_model, bounds=BOUNDS, lambda_=60)

    x_best, y_best, es_hist = es.evolute(ngen=50)

    best_case_dict = neorl_model.decode(x_best)
    print("Best Value:", y_best * neorl_model.sign)  # Flip sign back
    print("Best Inputs:", best_case_dict)

if 1==1:
    print("\n[Scipy] Running Optimization...")
    from scipy.optimize import differential_evolution
    
    bounds, cat_map = cgm.get_scipy_bounds(variable_inputs)
    scipy_model = cgm.ScipyWrapper( ff=model_Ackley, value_key=par_opt, variable_inputs=list(
        variable_inputs.keys()),  fixed_inputs=fixed_inputs, cat_map=cat_map,mode='minimize')

    result = differential_evolution(scipy_model, bounds)
    best_case = scipy_model.decode(result.x)

    print("Best Value:", result.fun * scipy_model.sign)  # Flip sign back if maximizing
    print("Best Inputs (Decoded):", best_case)
                                
                                

```




