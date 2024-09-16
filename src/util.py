import pandas as pd
import numpy as np
import time
import functools
import inspect
import os
import shutil
import datetime
import math
from os.path import join as pjoin


def timer(func):
    """Print the runtime of the decorated function"""

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()  # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()  # 2
        run_time = end_time - start_time  # 3
        # logging.info(f"Finished {func.__name__!r} in {run_time:.6f} secs")
        print(f"Finished {func.__name__!r} in {run_time:.6f} secs")
        return value

    return wrapper_timer


# class CustomSeries(pd.Series):
#     @property
#     def v(self):
#         return self


# class CustomDataFrame(pd.DataFrame):
#     @property
#     def v(self):
#         return self


# # Monkey-patch pandas DataFrame
# pd.DataFrame = CustomDataFrame
# pd.Series = CustomSeries
# pd.core.frame.DataFrame = CustomDataFrame
# pd.core.frame.Series = CustomSeries


def convert_to_float(value):
    try:
        return float(value)
    except ValueError:
        return value


def roundSF(x, sig=1):
    """
    Rounds a number to a specified number of significant figures
    :param x:
    :param sig:
    :return:
    """
    if x == 0:
        return 0
    # deal with arrays

    return round(x, sig - int(math.floor(math.log10(abs(x)))) - 1)


vectorized_roundSF = np.vectorize(roundSF)


def format_float(value, sig_figs=3, scientific_notation=False):
    if isinstance(value, (int, float)):
        if value == 0:
            return "0." + "0" * (sig_figs - 1)

        exponent = int(math.floor(math.log10(abs(value))))
        decimal_places = sig_figs - 1 - exponent

        format_specifier_e = "{{:,.{}e}}".format(sig_figs)
        value_e = float(format_specifier_e.format(value))

        if decimal_places < 0:
            format_specifier_f = "{{:,.{}f}}".format(0)
        else:
            format_specifier_f = "{{:,.{}f}}".format(decimal_places)

        if scientific_notation:
            formatted_value = format_specifier_e.format(value)
        else:
            formatted_value = format_specifier_f.format(value_e)
        return formatted_value

    # if list, tuple, or array
    elif isinstance(value, (list, tuple, np.ndarray)):
        return [
            format_float(v, sig_figs=sig_figs,
                         scientific_notation=scientific_notation)
            for v in value
        ]

    else:
        return value


def fn_eval_t(df_d, df_EX, fn):
    """
    Used to test functions and see what variables they generate
    :param df:
    :param df_EX:
    :param fn:
    :return:
    """
    old_vars = df_d.columns.values
    df_d, df_EX = fn(df_d, df_EX)
    new_vars = np.setdiff1d(df_d.columns.values, old_vars)
    print("New variables:")
    print(new_vars)
    print(df_d[new_vars].iloc[0].T)
    return df_d, df_EX


def fn_eval_test(df, fn):
    """
    Used to test functions and see what variables they generate
    :param df:
    :param fn:
    :return:
    """
    df = fn(df)
    print([[f, k.v] for f, k in df.__dict__.items() if k.Function == fn.__name__])

    return df


def fun_name():
    return inspect.currentframe().f_back.f_code.co_name


def get_directory_above_file(f):
    # Get the absolute path of the current file
    current_file_path = os.path.abspath(f)

    # Get the directory containing the current file
    current_directory = os.path.dirname(current_file_path)

    # Get the directory above the current directory
    directory_above = os.path.dirname(current_directory)

    return directory_above


def dfn_dec(func):
    """
    Decorator added to OMEGA14 functions
    :param func:
    :return:
    """

    @functools.wraps(func)
    def dfn_dec_wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()  # 2
        run_time = end_time - start_time  # 3
        # logging.info(f"Finished {func.__name__!r} in {run_time:.6f} secs")
        # print(f"Finished {func.__name__!r} in {run_time:.6f} secs")
        value[f"Time {func.__name__!r}"] = roundSF(run_time, 3)
        return value

    return dfn_dec_wrapper


def dfn_dict(value, unit="NA", description="NA", system="NA"):
    """Use to add entries to a design_df_class.

    :param value:
    :param unit:
    :param function:
    :param description:
    :return:
    """
    return {
        "v": value,
        "Unit": unit,
        "Function": inspect.currentframe().f_back.f_code.co_name,
        "Description": description,
        "System": system,
    }


def display_sigfig(x, sigfigs=2) -> str:
    """
    Suppose we want to show 2 significant figures. Implicitly we want to show 3 bits of information:
    - The order of magnitude
    - Significant digit #1
    - Significant digit #2
    """
    if np.isnan(x):
        return x
    else:
        if sigfigs < 1:
            raise Exception(
                "Cannot have fewer than 1 significant figures. ({} given)".format(
                    sigfigs
                )
            )

        order_of_magnitude = math.floor(math.log10(abs(x)))

        # Because we get one sigfig for free, to the left of the decimal
        decimals = sigfigs - 1

        x /= pow(10, order_of_magnitude)
        x = round(x, decimals)
        x *= pow(10, order_of_magnitude)

        # Subtract from decimals the sigfigs we get from the order of magnitude
        decimals -= order_of_magnitude
        # But we can't have a negative number of decimals
        decimals = max(0, decimals)

        return "{:,.{dec}f}".format(x, dec=decimals)


def custom_serializer(obj):
    if isinstance(obj, datetime.datetime):
        return obj.isoformat()
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, (np.ndarray, np.generic)):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    else:
        raise TypeError(f"Type {type(obj)} not serializable")


def create_dir_safe(folder):
    """
    Creates a directory if it does not exist.
    :param folder:
    :return:
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
        print("creating directory: " + folder)
    return


def create_dir(folder):
    """
    Creates a directory if it does not exist. Otherwise, it overwites.
    :param folder:
    :return:
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
        print("creating directory: " + folder)
    else:
        shutil.rmtree(folder)
        os.makedirs(folder)
        print("overwriting directory: " + folder)
    return


def df_to_dict(df,x):
    """

    Add a dataframe df to a dictionary x with the index_col as the key and the entries as the values.

    Parameters:
    df (pandas.DataFrame): The dataframe to be added to the dictionary.
    x (dict): The dictionary to add the dataframe entries to.

    Returns:
    dict: The updated dictionary with the dataframe entries added.
    """
    for index, row in df.to_dict(orient='index').items():
        for col, value in row.items():
            x[f"{index}_{col}"] = value
    return x


if __name__ == "__main__":

    if 1 == 1:

        par_space = {"A": np.linspace(0.1, 0.2, 3), "B": np.linspace(5, 6, 3)}

        d = generate_combos(par_space=par_space)

        print(d)

        par_space = {
            "x0": {
                "mean": 1,
                "unc": 0,
                "unc_type": "normal",
                "type": "float",
                "range": [1, 2],
                "bounds": [0, np.inf]
            },
            "x1": {"mean": 1, "unc": 1, "range": [0, 2], "unc_type": "uniform"},
            "x2": {
                "mean": 3,
                "unc": 0,
                "unc_type": "normal",
                "type": "int",
            },
            "x3": {
                "mean": 4,
                "unc": .1,
                "unc_type": "uniform",
                "type": "float",
            },
            "x4": {
                "mean": "ppp",
                "options": ["a"],
                "unc_type": "choice",
                "bounds": None,
                "type": "options",
                "unc": np.array([1.0, 1.0, 1.0]) / 3,
            },
        }

        par_space = process_input_stack(par_space)
        print(par_space)

        a = generate_samples(par_space, n=1000, type="grid")

        # generate_combos_rand({'a': (1, 5), 'b': [2, 4, 6]},n=10000)
        print(a)

    else:

        # Test fn_eval_t
        print("Testing fn_eval_t:")
        df_d = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        df_EX = pd.DataFrame({"C": [7, 8, 9]})

        def test_fn(df_d, df_EX):
            df_d["D"] = df_d["A"] + df_d["B"]
            return df_d, df_EX

        fn_eval_t(df_d, df_EX, test_fn)

        # Test fn_eval_test
        print("\nTesting fn_eval_test:")

        class TestDF:
            def __init__(self):
                self.A = type("obj", (), {"v": 1, "Function": "test_fn"})()
                self.B = type("obj", (), {"v": 2, "Function": "other_fn"})()

        df = TestDF()

        def test_fn(df):
            return df

        fn_eval_test(df, test_fn)

        # Test fun_name
        print("\nTesting fun_name:")

        def test_function():
            print(f"Current function name: {fun_name()}")

        test_function()

        # Test get_directory_above_file
        print("\nTesting get_directory_above_file:")
        print(
            f"Directory above current file: {get_directory_above_file(__file__)}")

        # Test check_input_valid
        print("\nTesting check_input_valid:")
        base_inputs = {"a": 1, "b": 2, "c": 3}
        add_inputs = {"a": 4, "b": 5}
        try:
            check_input_valid(base_inputs, add_inputs)
            print("Valid inputs")
        except ValueError as e:
            print(f"Invalid inputs: {e}")

        # Test custom_serializer
        print("\nTesting custom_serializer:")
        test_data = {
            "datetime": datetime.datetime.now(),
            "set": {1, 2, 3},
            "numpy_array": np.array([1, 2, 3]),
            "dataframe": pd.DataFrame({"A": [1, 2], "B": [3, 4]}),
            "series": pd.Series([1, 2, 3]),
        }
        for key, value in test_data.items():
            print(f"{key}: {custom_serializer(value)}")
