import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import IntegerRandomSampling

# --- [Your existing setup code] ---
# (Assuming 'model', 'fixed_inputs', and 'variable_inputs' are defined as in your snippet)

# 1. Generate Pymoo bounds and Categorical Map
# Note: Ensure variable_inputs passed here is the dictionary, not just keys
xl, xu, cat_map = get_pymoo_bounds(variable_inputs)

# 2. Create the Pymoo Problem Instance
problem_class = create_pymoo_problem(
    ff=model,
    value_key="y0",
    variable_inputs=list(variable_inputs.keys()), # Pass list of names
    fixed_inputs=fixed_inputs,
    cat_map=cat_map
)

# Instantiate the problem with the calculated bounds
problem = problem_class(n_var=len(variable_inputs), xl=xl, xu=xu)

# 3. Setup the Algorithm
# We use DE (Differential Evolution) as a robust default.
# Note: If you have categorical variables, GA might be safer with integer sampling,
# but standard DE often works well if we round values inside the wrapper (which we do).
algorithm = DE(pop_size=20)

# 4. Run Optimization
res = minimize(problem,
               algorithm,
               seed=1,
               verbose=True)

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))

# 5. Decode the best solution back to human-readable format
decoded_solution = {}
for i, name in enumerate(variable_inputs.keys()):
    if i in cat_map:
        idx = int(np.round(res.X[i]))
        decoded_solution[name] = cat_map[i][idx]
    else:
        decoded_solution[name] = res.X[i]

print("Decoded Inputs:", decoded_solution)