import numpy as np


class NeorlWrapper:
    """
    A pickle-able wrapper class for NEORL optimization.
    """

    def __init__(self, ff, value_key, variable_inputs, fixed_inputs):
        self.ff = ff
        self.value_key = value_key
        self.variable_inputs = variable_inputs
        self.fixed_inputs = fixed_inputs

    def __call__(self, x):
        # 1. Map the list 'x' (from NEORL) to the variable names
        mapped_inputs = {name: x[i] for i, name in enumerate(self.variable_inputs)}

        # 2. Merge with fixed inputs
        all_inputs = {**mapped_inputs, **self.fixed_inputs}

        # 3. Call the original function
        output = self.ff(all_inputs)

        # 4. Return the specific fitness value
        return output[self.value_key]


def create_NEORL_funwrap(ff, value_key, variable_inputs, fixed_inputs):
    """
    Factory function to create the NeorlWrapper instance.
    """
    return NeorlWrapper(ff, value_key, variable_inputs, fixed_inputs)


def NEORL_getbounds(input_stack):
    """
    Generates the bounds dictionary required by NEORL (x1, x2, ...).
    """
    i_B = 1
    BOUNDS = {}

    for key, value in input_stack.items():
        if isinstance(value, dict):

            # Case A: Categorical / Grid
            if "options" in value:
                BOUNDS['x' + str(i_B)] = ['grid', tuple(value['options'])]

            # Case B: Continuous (Float/Int)
            else:
                # Determine type (default to float)
                var_type = value.get('type', 'float')

                # specific handling for min/max vs bounds vs range
                if 'bounds' in value:
                    lb, ub = value['bounds'][0], value['bounds'][1]
                elif 'range' in value:
                    lb, ub = value['range'][0], value['range'][1]
                elif 'min' in value and 'max' in value:
                    lb, ub = value['min'], value['max']
                else:
                    raise ValueError(f"Variable '{key}' needs 'bounds', 'range', or 'min'/'max'.")

                BOUNDS['x' + str(i_B)] = [var_type, lb, ub]

            i_B += 1

    return BOUNDS


class ScipyWrapper:
    """
    A pickle-able wrapper class for Scipy optimization.
    Handles mapping continuous Scipy variables back to named/categorical inputs.
    """

    def __init__(self, ff, value_key, variable_inputs, fixed_inputs, cat_map):
        self.ff = ff
        self.value_key = value_key
        self.variable_inputs = variable_inputs
        self.fixed_inputs = fixed_inputs
        self.cat_map = cat_map

    def __call__(self, x):
        # 1. Reconstruct the dictionary input for the model
        param_dict = self.fixed_inputs.copy()

        for i, name in enumerate(self.variable_inputs):
            if i in self.cat_map:
                # Handle categorical/discrete variables
                # Scipy provides a float; we round to nearest index
                idx = int(np.round(x[i]))
                # Safety clip to ensure index is valid
                idx = max(0, min(idx, len(self.cat_map[i]) - 1))
                param_dict[name] = self.cat_map[i][idx]
            else:
                # Continuous variables
                param_dict[name] = x[i]

        # 2. Call the original function
        try:
            # We assume the user wants to MAXIMIZE the value in 'value_key'.
            # Scipy minimizes, so we return negative fitness.
            output = self.ff(param_dict)
            val = output[self.value_key]
            return -val
        except Exception as e:
            # Return a large penalty if model fails
            return 1e12


def create_scipy_funwrap(ff, value_key, variable_inputs, fixed_inputs, cat_map=None):
    """
    Factory function to create the ScipyWrapper instance.
    """
    if cat_map is None:
        cat_map = {}
    return ScipyWrapper(ff, value_key, variable_inputs, fixed_inputs, cat_map)


def get_scipy_bounds(input_stack):
    """
    Generates bounds list for Scipy DE and a map for categorical variables.

    Returns:
        bounds (list of tuples): [(min, max), ...]
        cat_map (dict): {index: [option1, option2, ...]} for discrete vars.
    """
    bounds = []
    cat_map = {}

    # Iterate over keys to maintain order corresponding to the 'x' array
    for i, (key, value) in enumerate(input_stack.items()):

        # Case A: Categorical / Discrete Options
        if "options" in value:
            options = value["options"]
            # The optimizer sees a continuous range representing indices [0, len-1]
            bounds.append((0, len(options) - 1))
            cat_map[i] = options

        # Case B: Continuous Variables
        else:
            # Support multiple keywords for bounds definition
            if 'bounds' in value:
                low, high = value['bounds'][0], value['bounds'][1]
            elif 'range' in value:
                low, high = value['range'][0], value['range'][1]
            elif 'min' in value and 'max' in value:
                low, high = value['min'], value['max']
            else:
                # Fallback or error if no bounds found
                raise ValueError(f"Variable '{key}' must have 'bounds', 'range', or 'options' defined.")

            bounds.append((low, high))

    return bounds, cat_map