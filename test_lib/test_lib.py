import time
import numpy as np


def test_time(func_name, call_num, func, args):
    start_time = time.time()
    for i in range(call_num):
        results = func(**args)
    print(f"{func_name} time = {time.time() - start_time}")
    return results


def compare_functions(test_name, compare_dict, call_num, verbose=False):
    print(f'------------ {test_name} ------------')
    results_dict = {}
    # run time tests
    for key, value in compare_dict.items():
        results_dict[key] = {}
        results_dict[key]['res'] = test_time(func_name=key, call_num=call_num, **compare_dict[key])
    print("-----COMPARE RESULTS--------")
    # round results for equal tests
    for key in results_dict.keys():
        results_dict[key]['res'] = np.round(results_dict[key]['res'], 3)

    for i in range(len(results_dict.keys()) - 1):
        curr_key = list(results_dict.keys())[i]
        next_key = list(results_dict.keys())[i + 1]
        if (results_dict[curr_key]['res'] == results_dict[next_key]['res']).all():
            print(f'results {curr_key} == {next_key}')

        else:
            print(f'results {curr_key} != {next_key}')
            if verbose:
                print(f"\nkernel {curr_key} =\n {str(results_dict[curr_key]['res'])}")
                print(f"\nkernel {next_key} =\n {str(results_dict[next_key]['res'])}")
        print('--')
