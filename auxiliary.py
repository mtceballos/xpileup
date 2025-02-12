import math

def vprint(*args, **kwargs):
    """
    Print function that can be turned on/off with the verbose variable.
    """
    verbose = kwargs.pop("verbose", 1)
    if verbose > 0:
        print(*args, **kwargs)


def time_to_observe_n_counts(rate, time_interval, n):
    """
    Calculate the expected time to observe n counts in a given time interval under Poisson statistics.

    Parameters:
        rate (float): The count rate of the source (counts per second).
        time_interval (float): The time interval (in seconds) during which we want n counts.
        n (int): The number of counts to observe.

    Returns:
        float: The expected observation time (in seconds) to see n counts in the given time interval.
    """
    if n <= 0:
        raise ValueError("The number of counts (n) must be a positive integer.")

    # Calculate the expected number of counts in the time interval
    mu = rate * time_interval

    # Calculate the probability of observing n counts using the Poisson formula
    p_n = (mu ** n * math.exp(-mu)) / math.factorial(n)

    # Calculate the rate of observing n counts
    rate_of_n_counts = p_n / time_interval

    # Return the expected observation time (reciprocal of the rate)
    return 1 / rate_of_n_counts