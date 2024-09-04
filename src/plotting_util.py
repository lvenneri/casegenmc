import math
import numpy as np
def generate_xticks(x_values= None, min_value=None, max_value= None, num_ticks=None, precision=10):

    # Find the minimum and maximum x-values
    if x_values is not None:
        min_value = min(x_values)
        max_value = max(x_values)
    else:

        x_values =np.linspace(min_value, max_value, 10)

    def round_to_nearest(number, base):
        return round(number / base) * base

    def round_to_precision(number, precision):
        return round(number, precision)

    def calculate_round_spacing(value):
        order_of_magnitude = 10 ** math.floor(math.log10(value))
        scaled_value = value / order_of_magnitude

        if scaled_value < 2.5:
            return 1 * order_of_magnitude
        elif scaled_value < 7.5:
            return 5 * order_of_magnitude
        else:
            return 10 * order_of_magnitude



    # Adaptively pick the number of ticks if not specified
    if num_ticks is None:
        num_ticks = round(math.sqrt(len(x_values)))

    # Calculate the range and tick interval
    value_range = max_value - min_value
    tick_interval = value_range / (num_ticks - 1)

    # Determine the best round spacing for the tick interval
    rounded_tick_interval = calculate_round_spacing(tick_interval)

    # Adjust the number of ticks based on the new tick interval
    num_ticks = math.ceil(value_range / rounded_tick_interval) + 1

    # If the range covers 0, ensure that 0 is one of the x-ticks
    if min_value <= 0 <= max_value:
        starting_tick = -math.floor(-min_value / rounded_tick_interval) * rounded_tick_interval
    else:
        starting_tick = round_to_nearest(min_value, rounded_tick_interval)

    # Calculate the x-ticks by adding the rounded tick interval iteratively
    x_ticks = [round_to_precision(starting_tick + i * rounded_tick_interval, precision) for i in range(num_ticks)]

    return x_ticks

