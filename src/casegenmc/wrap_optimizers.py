import numpy as np

class NeorlWrapper:
    """
    A wrapper class for NEORL optimization.

    Features:
    - Decodes NEORL vectors back to dictionary inputs.
    - Handles Maximization vs Minimization (standardizes on Minimization).
    - Robust error handling.
    """

    def __init__(self, ff, value_key, variable_inputs, fixed_inputs, mode='minimize'):
        """
        :param ff: The model function to evaluate.
        :param value_key: The dictionary key of the result to optimize.
        :param variable_inputs: List of variable names in the order NEORL sees them.
        :param fixed_inputs: Dictionary of constant parameters.
        :param mode: 'maximize' (default) or 'minimize'.
        """
        self.ff = ff
        self.value_key = value_key
        self.variable_inputs = variable_inputs
        self.fixed_inputs = fixed_inputs

        if mode not in ['maximize', 'minimize']:
            raise ValueError("mode must be 'maximize' or 'minimize'")

        # Strategy: We convert everything to a Minimization problem.
        # If the user wants to MAXIMIZE, we return -Value.
        # This allows you to ALWAYS set the NEORL optimizer to mode='min'.
        self.sign = -1.0 if mode == 'maximize' else 1.0

    def decode(self, x):
        """
        Converts the raw NEORL list 'x' back into the full dictionary
        expected by the model, including fixed inputs.
        """
        # Map the list 'x' (from NEORL) to the variable names
        mapped_inputs = {name: x[i] for i, name in enumerate(self.variable_inputs)}

        # Merge with fixed inputs
        all_inputs = {**mapped_inputs, **self.fixed_inputs}

        return all_inputs

    def __call__(self, x):
        inputs = self.decode(x)
        try:
            output = self.ff(inputs)
            val = output[self.value_key]
            return val * self.sign

        except Exception:
            return 1e12


def create_NEORL_funwrap(ff, value_key, variable_inputs, fixed_inputs, mode='maximize'):
    return NeorlWrapper(ff, value_key, variable_inputs, fixed_inputs, mode)


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
    A wrapper class for Scipy optimization.
    Now includes a helper to decode raw results back to model inputs.
    """

    def __init__(self, ff, value_key, variable_inputs, fixed_inputs, cat_map=None, mode='maximize'):
        self.ff = ff
        self.value_key = value_key
        self.variable_inputs = variable_inputs
        self.fixed_inputs = fixed_inputs
        self.cat_map = cat_map if cat_map is not None else {}

        if mode not in ['maximize', 'minimize']:
            raise ValueError("mode must be 'maximize' or 'minimize'")
        self.sign = -1.0 if mode == 'maximize' else 1.0

    def decode(self, x):
        """
        Converts the raw optimizer array 'x' back into the full dictionary
        expected by the model, including fixed inputs.
        """
        # Start with fixed inputs
        param_dict = self.fixed_inputs.copy()

        # Map variable inputs
        for i, name in enumerate(self.variable_inputs):
            if i in self.cat_map:
                # Handle categorical/discrete variables
                # Round float to nearest index and clip to bounds
                idx = int(np.round(x[i]))
                idx = max(0, min(idx, len(self.cat_map[i]) - 1))
                param_dict[name] = self.cat_map[i][idx]
            else:
                # Continuous variables
                param_dict[name] = x[i]

        return param_dict

    def __call__(self, x):
        # Use the internal decode method to get parameters
        param_dict = self.decode(x)

        try:
            output = self.ff(param_dict)
            val = output[self.value_key]
            return val * self.sign
        except Exception:
            return 1e12


def create_scipy_funwrap(ff, value_key, variable_inputs, fixed_inputs, cat_map=None, mode='maximize'):
    return ScipyWrapper(ff, value_key, variable_inputs, fixed_inputs, cat_map, mode)


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