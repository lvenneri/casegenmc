# CASEGEN MC


Probe a model and see the possibilities. Takes model and input dictionary and evaluates cases to explore the model space using grids, random sampling, and optimization techniques.

The input dictionary is defined by the user with mean value, uncertainty, uncertainty distribution, range, and bounds. Sparse definition okay and assumes 0 unc and range by default. Numerical and categorical parameters are supported.

Includes matplotlib utility functions for standard plotting.


| Analysis Type                    | Description                                                                                      |
|----------------------------|--------------------------------------------------------------------------------------------------|
| `estimate`                 | Runs the model with the mean values of the input parameters.                                    |
| `estimate_with_unc`        | Runs the model with sampled input parameters based on their uncertainty distributions.          |
| `estimate_with_unc_combos` | Runs the model with combinations of extreme values of the input parameters.                     |
| `sensitivity_analysis_unc` | Performs sensitivity analysis by varying each parameter individually based on its uncertainty distribution. |
| `sensitivity_analysis_range` | Performs sensitivity analysis by varying each parameter individually over its entire range.    |
| `sensitivity_analysis_2D`  | Performs 2D sensitivity analysis by varying two parameters simultaneously over a grid.         |
| `regular_grid`             | Runs the model over a regular grid of input parameter values.                                   |
| `random_uniform_grid`      | Runs the model over a grid of randomly sampled input 


## Install
```
pip install casegenmc
```


## Use
```python
import casegenmc as cgm 

# Define model
def model(x):
    out = {}
    out["y0"] = x["x0"]**2 + np.exp(x["x1"]) + x['x3']
    out["y1"] = x["x0"] + x["x1"] + x["x2"] + x["x3"]
    return out

# Create input stack. some parameters fixed - don't have uncertainty or range of options.
# mean, unc, range, bounds (minimum and maximum value), unc_type
input_stack = {
    "x0": {"mean": 1., "unc": .2, 'range': [0, 5], 'bounds': [0, 100], 'unc_type': 'normal'},
    "x1": {"mean": 1., "unc": .2, 'range': [0, 3], 'unc_type': 'lognormal'},
    "x2": 3., 
    "x3": 4, 
    "x4": {"mean": "a",  'range': ["a", "b"], "options": ["a", "b", "c"], "unc_type": "choice", },
    }

# Pre-process input stack
input_stack = cgm.process_input_stack(input_stack)
print(input_stack)

# Evaluate the model with the input_stack.
cgm.run_analysis(model=model, input_stack=input_stack, analyses=["estimate"])

# Estimate with uncertainty.
cgm.run_analysis(model, input_stack, n_samples=1000, analyses=["estimate_with_unc"], par_output="y0")  

# Estimate with uncertainty combinations.
cgm.run_analysis(model, input_stack, n_samples=1000, analyses=["estimate_with_unc_combos"], par_output="y0")  

# 2d sensitivity analysis and analysis w.r.t 1 output parameter.
cgm.run_analysis(model, input_stack, n_samples=1000, analyses=["sensitivity_analysis_2D"],  par_grid_xy=["x0", "x1"], par_output="y0")

# sample a regular grid defined by range.
cgm.run_analysis(model, input_stack, n_samples=1000, analyses=["regular_grid"],  par_output="y0")


```



