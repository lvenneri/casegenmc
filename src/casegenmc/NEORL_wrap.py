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