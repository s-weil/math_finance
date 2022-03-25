import numpy as np
from scipy.stats import norm


def simulate_paths(drift, vola, price_0, nr_steps, nr_paths):
    zs = norm.ppf(np.random.rand(nr_steps, nr_paths)) 
    daily_returns = np.exp(drift + vola * zs)
    price_paths = np.zeros_like(daily_returns)
    price_paths[0] = price_0
    for t in range(1, nr_steps):
        price_paths[t] = price_paths[t-1]*daily_returns[t]
    return price_paths



def profile(func):
    import timeit
    nr = 100

    def wrapper(): #*args, **kwargs):
        print("starting")
        func() #args, **kwargs)
        print((timeit.timeit(func, number=nr)) / nr)

    return wrapper


@profile
def profile_gbm():
    simulate_paths(0.001, 0.005, 100.0, 100, 10_000)


if __name__ == '__main__':

    profile_gbm()

    import cProfile
    cProfile.run('my_func()', 'output.dat')

    import pstats
    from pstats import SortKey

    with open("output_time.txt", "w") as f:
        p = pstats.Stats("output.dat", stream = f)
        p.sort_stats("time").print_stats()

    with open("output_calls.txt", "w") as f:
        p = pstats.Stats("output.dat", stream = f)
        p.sort_stats("calls").print_stats()

# https://medium.com/swlh/4-simple-libraries-to-quickly-benchmark-python-code-8d3dfd288d7a
# add import cProfile
# run: 
# python -m cProfile -o results.prof main.py
