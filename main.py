import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

COMPLICATED_TIME_PROBABILITY = 0.25
CLEAN_TIME = 20
SIMPLE_LOWER_LIMIT = 40 + CLEAN_TIME
SIMPLE_UPPER_LIMIT = 60 + CLEAN_TIME
COMPLICATED_LOWER_LIMIT = 120 + CLEAN_TIME
COMPLICATED_UPPER_LIMIT = 180 + CLEAN_TIME


def generate_surgery_time(num_of_samples, num_of_patients):
    chances = np.random.binomial(n=1, p=COMPLICATED_TIME_PROBABILITY,
                                 size=(num_of_samples, num_of_patients))
    simple = np.random.uniform(SIMPLE_LOWER_LIMIT, SIMPLE_UPPER_LIMIT,
                               size=(num_of_samples, num_of_patients))
    complicated = np.random.uniform(COMPLICATED_LOWER_LIMIT, COMPLICATED_UPPER_LIMIT,
                                    size=(num_of_samples, num_of_patients))
    return np.where(chances, complicated, simple)


WAIT_COST = 1000
OVERTIME_COST = 3000
REVENUE = 3500
TOTAL_TIME = 540


def calculate_profit(num_of_samples, num_of_patients, shift_time):
    start_time = np.zeros(num_of_samples)             # in minutes
    total_waiting_time = np.zeros(num_of_samples)     # in minutes
    finish_time = np.zeros(num_of_samples)            # in minutes
    actual_times = generate_surgery_time(num_of_samples, num_of_patients)
    for i in range(num_of_patients):
        scheduled_start_time = i * shift_time
        wait_time = np.maximum(start_time - scheduled_start_time, 0)
        total_waiting_time += wait_time
        actual_time = actual_times[:, i]
        finish_time = actual_time + start_time
        scheduled_finish_time = (i + 1) * shift_time
        start_time = np.maximum(finish_time, scheduled_finish_time)
    overtime = np.maximum(finish_time - TOTAL_TIME, 0)
    return REVENUE * num_of_patients \
        - OVERTIME_COST * overtime / 60 \
        - WAIT_COST * total_waiting_time / 60


MIN_NUM_OF_PATIENTS = 1
MAX_NUM_OF_PATIENTS = 10
MIN_SHIFT_TIME = 20
MAX_SHIFT_TIME = 240
NUM_OF_SAMPLES = 10000


def main():
    # --- Calculate ---
    # When num_of_patients = 6, shift_time = 95
    # 100000 samples will need about 3.4s to calculate without numpy
    # 10000000 samples will need about 3.3s to calculate with numpy
    start_time = time.time()
    numbers_of_patients = np.arange(MIN_NUM_OF_PATIENTS, MAX_NUM_OF_PATIENTS + 1)
    shift_times = np.arange(MIN_SHIFT_TIME, MAX_SHIFT_TIME + 1)
    average_profits = np.empty((shift_times.shape[0], numbers_of_patients.shape[0]))
    lower_95_ci = np.empty_like(average_profits)
    upper_95_ci = np.empty_like(average_profits)
    t_critical = stats.t.ppf(q=.975, df=NUM_OF_SAMPLES - 1)
    for i, iv in enumerate(numbers_of_patients):
        for j, jv in enumerate(shift_times):
            profits = calculate_profit(NUM_OF_SAMPLES, iv, jv)
            avg_profit = np.mean(profits)
            average_profits[j, i] = avg_profit
            stddev = np.std(profits, ddof=1)
            standard_error = stddev / np.sqrt(NUM_OF_SAMPLES)
            lower_95_ci[j, i] = avg_profit - t_critical * standard_error
            upper_95_ci[j, i] = avg_profit + t_critical * standard_error
    print('--- %s seconds ---' % (time.time() - start_time))
    # --- Find the biggest ---
    i, j = np.unravel_index(average_profits.argmax(), average_profits.shape)
    print('Best case: patients: %d, shift time: %d' % (numbers_of_patients[j], shift_times[i]))
    # --- Export CSV ---
    writer = pd.ExcelWriter('result.xlsx', engine='xlsxwriter')
    pd.DataFrame(average_profits, index=shift_times, columns=numbers_of_patients) \
        .to_excel(writer, sheet_name='Average Profit')
    pd.DataFrame(lower_95_ci, index=shift_times, columns=numbers_of_patients) \
        .to_excel(writer, sheet_name='Lower 95 CI of Profit')
    pd.DataFrame(upper_95_ci, index=shift_times, columns=numbers_of_patients) \
        .to_excel(writer, sheet_name='Higher 95 CI of Profit')
    writer.save()
    # --- Plot ---
    x, y = np.meshgrid(numbers_of_patients, shift_times)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(x, y, average_profits)
    ax.set_xlabel('Number of Patients')
    ax.set_ylabel('Shift Time')
    ax.set_zlabel('Profit')
    plt.show()


if __name__ == '__main__':
    main()
