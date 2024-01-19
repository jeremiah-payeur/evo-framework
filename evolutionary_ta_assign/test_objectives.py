import pandas as pd
import ta_assign_evo as ta

# load data
tas = pd.read_csv('tas.csv')
sections = pd.read_csv('sections.csv')
W = ta.pd_numpy(tas, 3, 20)
min = sections['min_ta'].to_numpy()
time = sections['daytime'].to_numpy()
max = tas['max_assigned'].to_numpy()

def create_evo():
    E = ta.Evo()
    E.add_fitness_criteria('overallocation', ta.overallocation, max)
    E.add_fitness_criteria('conflicts', ta.conflicts, time)
    E.add_fitness_criteria('under', ta.under_support, min)
    E.add_fitness_criteria('unwilling', ta.unwilling, W)
    E.add_fitness_criteria('unpreffered', ta.unpreffered, W)
    return E

def run_test(test_file, expected_result):
    test = pd.read_csv(test_file, header=None)
    E = create_evo()
    E.add_solution(test.to_numpy())
    assert list(E.pop.keys())[0] == expected_result

def test_1():
    expected_result = (('overallocation', 37), ('conflicts', 8), ('under', 1), ('unwilling', 53), ('unpreffered', 15))
    run_test('test1.csv', expected_result)

def test_2():
    expected_result = (('overallocation', 41), ('conflicts', 5), ('under', 0), ('unwilling', 58), ('unpreffered', 19))
    run_test('test2.csv', expected_result)

def test_3():
    expected_result = (('overallocation', 23), ('conflicts', 2), ('under', 7), ('unwilling', 43), ('unpreffered', 10))
    run_test('test3.csv', expected_result)
