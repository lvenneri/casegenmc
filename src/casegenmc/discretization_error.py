import numpy as np
import copy
import casegenmc as cgm

def est_discretization_err(model, input_stack, grid_variable, key_variables, grid_sizes=None,
                           refinement_factor=2.0):
    """
    Estimates discretization error using Richardson Extrapolation (RE) and the Grid Convergence Index (GCI).

    ASME Fluids Engineering Division: "Procedure for Estimation and Reporting of Uncertainty Due to Discretization in CFD Applications".

    Requires solutions from three significantly different grids (fine, medium, coarse) to
     estimate the apparent order of accuracy (p) and the uncertainty of the fine-grid solution.

    Parameters
    ----------
    model : function
        The model function to evaluate. Must accept a dictionary of inputs.
    input_stack : dict
        The base input parameters for the model.
    grid_variable : str
        The dictionary key in `input_stack` representing the representative grid size 'h'.
    key_variables : list of str
        The output variables to analyze for grid convergence (e.g., drag, lift, velocity).
    grid_sizes : list of float, optional
        A list of exactly 3 grid sizes. If provided, they will be sorted such that
        h1 < h2 < h3. If None, the current value in `input_stack` is treated as h1 (fine),
        and coarser grids are generated using `refinement_factor`.
    refinement_factor : float, optional
        The ratio between grid sizes (r = h_coarse / h_fine). Used if `grid_sizes` is None.
        ASME recommends r > 1.3. Default is 2.0.

    Returns
    -------
    dict
        A dictionary where keys are `key_variables` and values are statistics dictionaries containing:
        - 'phi_values': List of values [phi1, phi2, phi3] (fine to coarse).
        - [cite_start]'p': Apparent order of accuracy[cite: 57].
        - [cite_start]'phi_ext': Extrapolated "exact" value (Richardson Extrapolation)[cite: 75].
        - [cite_start]'ea': Approximate relative error[cite: 80].
        - [cite_start]'e_ext': Extrapolated relative error[cite: 81].
        - [cite_start]'GCI_fine': Fine-grid convergence index (uncertainty estimate)[cite: 82].
        - 'convergence_type': 'monotonic', 'oscillatory', or 'divergent'.
    """
    # 1. Setup Grid Sizes (h1 < h2 < h3)
    if grid_sizes is None:
        if isinstance(input_stack[grid_variable], dict):
            h1 = input_stack[grid_variable]['mean']
        else:
            h1 = input_stack[grid_variable]
        h2 = h1 * refinement_factor
        h3 = h2 * refinement_factor
        grids = [h1, h2, h3]
    else:
        grids = sorted(grid_sizes)

    h1, h2, h3 = grids
    r21 = h2 / h1
    r32 = h3 / h2

    print(f"--- Grid Convergence Study: {grid_variable} ---")
    print(f"Grids: {h1:.4f} (Fine) < {h2:.4f} (Medium) < {h3:.4f} (Coarse)")
    print(f"Refinement Factors: r21={r21:.2f}, r32={r32:.2f}")

    # 2. Run Model
    phi = {k: [] for k in key_variables}
    for h in grids:
        current_stack = copy.deepcopy(input_stack)
        if isinstance(current_stack[grid_variable], dict):
            current_stack[grid_variable]['mean'] = h
        else:
            current_stack[grid_variable] = h

        # Extract simple inputs for model
        simple_inputs = {k: (v['mean'] if isinstance(v, dict) else v) for k, v in current_stack.items()}
        res = model(simple_inputs)
        for k in key_variables:
            phi[k].append(res[k])

    # 3. Calculate Stats
    results = {}
    for k in key_variables:
        phi1, phi2, phi3 = phi[k]  # Fine, Med, Coarse
        eps21 = phi2 - phi1
        eps32 = phi3 - phi2

        stats = {"phi_fine": phi1, "phi_med": phi2, "phi_coarse": phi3}

        # --- CORRECTED CONVERGENCE CHECK ---
        # R is the convergence ratio
        if eps32 == 0:
            stats["convergence"] = "Steady (No Change)"
            R = 0
        else:
            R = eps21 / eps32

        if R < 0:
            stats["convergence"] = "Oscillatory"
        elif R > 1:
            # The difference is growing as we refine (Fine delta > Coarse delta)
            stats["convergence"] = "Divergent"
        else:
            # 0 < R < 1: The difference is shrinking (Fine delta < Coarse delta)
            stats["convergence"] = "Monotonic"

        # Calculate Order p (Iterative)
        if stats["convergence"] == "Monotonic" and eps21 != 0:
            s = 1 if (eps32 / eps21) > 0 else -1
            try:
                p = 1 / np.log(r21) * abs(np.log(abs(eps32 / eps21)))
                for _ in range(100):  # Fixed point iteration
                    q = np.log((r21 ** p - s) / (r32 ** p - s)) if (r21 ** p - s) / (r32 ** p - s) > 0 else 0
                    p_new = (1 / np.log(r21)) * abs(np.log(abs(eps32 / eps21)) + q)
                    if abs(p - p_new) < 1e-4: break
                    p = p_new
                stats["p_order"] = p

                # Extrapolated Value
                stats["phi_ext"] = (r21 ** p * phi1 - phi2) / (r21 ** p - 1)

                # Error Estimates
                stats["e_approx_rel"] = abs((phi1 - phi2) / phi1)
                stats["e_ext_rel"] = abs((stats["phi_ext"] - phi1) / stats["phi_ext"])
                stats["GCI_fine"] = (1.25 * stats["e_approx_rel"]) / (r21 ** p - 1)
            except:
                stats["note"] = "Calculation Failed"

        results[k] = stats
    return results



if __name__ == "__main__":
    # --- Basic Model & Input Stack ---

    def numerical_simulation_dummy(inputs):
        """
        Simulates a CFD solver where error depends on grid size 'h'.
        """
        h = inputs['h']

        # Variable 1: "Velocity" (1st Order Convergence)
        # Exact solution = 50.0. Error term = 10 * h
        # As h -> 0, y -> 50.0
        vel = 50.0 + 10.0 * h

        # Variable 2: "Pressure" (2nd Order Convergence)
        # Exact solution = 100.0. Error term = 20 * h^2
        # As h -> 0, y -> 100.0
        press = 100.0 + 20.0 * (h ** 2)

        # Variable 3: "Turbulence" (Noisy/Oscillatory)
        # Just for demonstration
        turb = 0.5 + 0.1 * np.sin(1 / h)

        return {"velocity": vel, "pressure": press, "turb": turb}


    # Define Inputs
    # We set 'h' to the finest grid size we intend to use.
    # The tool will generate coarser grids automatically.
    input_stack = {
        "h": 0.05,  # Fine Grid Size (e.g. meters)
        "inlet_temp": 300.0,  # Constant parameter
        "density": 1.225  # Constant parameter
    }

    # --- 3. Run Analysis ---

    # We analyze 'velocity' and 'pressure'
    # using a refinement factor of 2.0 (grid doubles in size each step)
    results = est_discretization_err(
        model=numerical_simulation_dummy,
        input_stack=input_stack,
        grid_variable="h",
        key_variables=["velocity", "pressure", "turb"],
        refinement_factor=2.0
    )

    # --- 4. Pretty Print Results ---
    for var, data in results.items():
        print(f"\nResults for '{var}':")
        print(f"  Convergence Type: {data.get('convergence')}")

        if 'p_order' in data:
            print(f"  Apparent Order (p): {data['p_order']:.2f}")
            print(f"  Fine Grid Value:    {data['phi_fine']:.4f}")
            print(f"  Extrapolated Value: {data['phi_ext']:.4f}")
            print(f"  GCI (Fine Grid):    {data['GCI_fine'] * 100:.2f}%")