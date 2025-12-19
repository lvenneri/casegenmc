import numpy as np
from pymoo.core.problem import ElementwiseProblem


def create_pymoo_problem(ff, value_key, variable_inputs, fixed_inputs, cat_map=None):
    """
    Creates a Problem object compatible with Pymoo.

    Parameters:
    - ff: The original function to be optimized (accepts a dict).
    - value_key: The key in the output dictionary of ff to minimize.
    - variable_inputs: A list of names for the variable inputs (order matters).
    - fixed_inputs: Dictionary of fixed parameters.
    - cat_map: (Optional) Dictionary mapping input indices to lists of options
               for categorical variables. e.g. {0: ['a', 'b', 'c']}.

    Returns:
    - An instance of a Pymoo ElementwiseProblem.
    """

    # We define the class dynamically or use a closure to capture the parameters
    class WrappedProblem(ElementwiseProblem):
        def __init__(self, n_var, xl, xu, **kwargs):
            super().__init__(n_var=n_var, n_obj=1, n_constr=0, xl=xl, xu=xu, **kwargs)

        def _evaluate(self, x, out, *args, **kwargs):
            mapped_inputs = {}

            for i, name in enumerate(variable_inputs):
                val = x[i]

                # If this variable is categorical, map the integer index back to the string/value
                if cat_map and i in cat_map:
                    # Ensure index is within bounds (handle floating point drift)
                    idx = int(np.clip(np.round(val), 0, len(cat_map[i]) - 1))
                    mapped_inputs[name] = cat_map[i][idx]
                else:
                    mapped_inputs[name] = val

            # Merge variable and fixed inputs
            all_inputs = {**mapped_inputs, **fixed_inputs}

            # Call original function
            output = ff(all_inputs)

            # Pymoo expects minimization by default.
            # If you need maximization, multiply by -1 here.
            out["F"] = output[value_key]

    return WrappedProblem


def get_pymoo_bounds(input_stack):
    """
    Extracts bounds and categorical mappings for Pymoo.
    """
    xl = []
    xu = []
    cat_map = {}  # Maps index in 'x' to list of options ['a', 'b']

    # Pymoo relies on order, so we iterate to ensure the order matches variable_inputs
    # (Assuming input_stack passed here only contains the variable inputs in correct order)

    for i, (key, value) in enumerate(input_stack.items()):

        # Check if it has 'options' (Categorical/Grid)
        if isinstance(value, dict) and "options" in value:
            # For Pymoo, we treat categorical choices as Integers [0, len(options)-1]
            options = value['options']
            xl.append(0)
            xu.append(len(options) - 1)
            cat_map[i] = options

        # Check if standard dictionary definition with 'bounds' or 'range'
        elif isinstance(value, dict):
            # Prioritize 'bounds' if it exists, else 'range'
            bounds = value.get('bounds', value.get('range'))
            xl.append(bounds[0])
            xu.append(bounds[1])

        # Fallback if value is just a list/tuple [min, max]
        elif isinstance(value, (list, tuple)):
            xl.append(value[0])
            xu.append(value[1])

    return np.array(xl), np.array(xu), cat_map