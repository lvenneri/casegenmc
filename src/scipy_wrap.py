import numpy as np


def create_scipy_funwrap(ff, value_key, variable_inputs, fixed_inputs, cat_map=None):
    """
    Creates a function compatible with scipy.optimize.

    Parameters:
    - ff: The original function to be optimized (accepts a dict).
    - value_key: The key in the output dictionary of ff to minimize.
    - variable_inputs: A list of names for the variable inputs (order matters).
    - fixed_inputs: Dictionary of fixed parameters.
    - cat_map: (Optional) Dictionary mapping input indices to lists of options
               for categorical variables. e.g. {0: ['a', 'b', 'c']}.

    Returns:
    - A function that accepts an array `x` and returns a float.
    """

    def scipy_compatible_function(x):
        mapped_inputs = {}

        # Iterate over x (the array SciPy provides) and the variable names
        for i, name in enumerate(variable_inputs):
            val = x[i]

            # If this variable index is in our categorical map, decode it
            if cat_map and i in cat_map:
                # 1. Round the continuous SciPy value to the nearest integer index
                idx = int(np.round(val))

                # 2. Safety clamp: ensure index doesn't go out of list bounds
                # (e.g. if SciPy tries 2.00000001 for a list of length 3)
                idx = max(0, min(idx, len(cat_map[i]) - 1))

                # 3. Retrieve the string option
                mapped_inputs[name] = cat_map[i][idx]
            else:
                # It's a regular float variable
                mapped_inputs[name] = val

        # Merge variable and fixed inputs
        all_inputs = {**mapped_inputs, **fixed_inputs}

        # Call original function
        output = ff(all_inputs)

        # Return the specific metric to minimize
        return output[value_key]

    return scipy_compatible_function


def get_scipy_bounds(input_stack):
    """
    Generates bounds and a categorical map for SciPy.

    Returns:
    - bounds: A list of tuples [(min, max), (min, max), ...]
    - cat_map: A dictionary {index: [options_list]} for decoding strings.
    """
    bounds = []
    cat_map = {}

    # Iterate through the inputs in order
    for i, (key, value) in enumerate(input_stack.items()):

        if isinstance(value, dict):
            # Check for Categorical / Options
            if "options" in value:
                # Map options to an integer range: [0, len(options)-1]
                # e.g., ["a", "b", "c"] -> Bounds (0, 2)
                opts = value['options']
                bounds.append((0, len(opts) - 1))
                cat_map[i] = opts

            # Check for Continuous Bounds
            elif "bounds" in value:
                bounds.append(tuple(value['bounds']))

            # Fallback for 'range' if 'bounds' is missing
            elif "range" in value:
                bounds.append(tuple(value['range']))

            # Fallback for type definition without explicit range (defaulting if necessary)
            else:
                # You might want to set default large bounds here if none exist
                bounds.append((None, None))

    return bounds, cat_map