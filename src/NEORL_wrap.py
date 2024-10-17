
def create_NEORL_funwrap(ff, value_key, variable_inputs, fixed_inputs):
    """
    Creates a function compatible with NEORL optimization package.

    Parameters:
    - ff: The original function to be optimized, which accepts a dictionary of parameters.
    - value_key: The key in the output dictionary of ff that is the output value to be returned.
    - variable_inputs: A list of names for the variable inputs, corresponding to the order in the array `x`.
    - fixed_inputs: Keyword arguments representing the fixed parameters for ff.

    Returns:
    - A function that accepts an array `x` of variable parameters and returns a single output.
    """
    def neorl_compatible_function(x):
  
        # Map the variable inputs array to the corresponding input names
        variable_inputs = {name: x[i]
                           for i, name in enumerate(variable_inputs)}

        # Merge variable and fixed inputs
        all_inputs = {**variable_inputs, **fixed_inputs}
        # Call the original function ff with the merged inputs
        output = ff(all_inputs)

        # Assuming ff returns a dictionary, select a single output value to return
        # Returns the first output value by default
        return output[value_key]

    return neorl_compatible_function



def NEORL_getbounds(input_stack):
    i_B = 1

    BOUNDS = {}

    for key, value in input_stack.items():

        # BOUNDS['x'+str(i_B)] = ['float', lb, ub]
        if isinstance(value, dict):

            if "options" in value:
                
                BOUNDS['x'+str(i_B)] = ['grid', tuple(input_stack[key]['options'])]
            else:

                BOUNDS['x'+str(i_B)] = [input_stack[key]['type'], input_stack[key]
                                        ['bounds'][0], input_stack[key]['bounds'][1]]
            i_B += 1

    return BOUNDS
