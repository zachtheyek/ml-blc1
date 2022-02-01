import pandas as pd
import csv


DATA_PREFIX = '/mnt_blpd2/datax/PKSUWL/blcxx/PKSUWL'

# META = pd.read_csv('first_11_meta_info.csv', sep=';')
# DRIFTS = pd.read_csv('first_11_drift_rates.csv', sep=';')
# CENTERS = pd.read_csv('first_11_start_frequencies.csv', sep=';')
META = pd.read_csv('lookalike_meta_info.csv', sep=';')
DRIFTS = pd.read_csv('lookalike_drift_rates.csv', sep=';')
CENTERS = pd.read_csv('lookalike_start_frequencies.csv', sep=';')

META['indexes'] = META['indexes'].str.strip()

with open('parameterization_inputs.csv') as f:
    csv_dict_reader = csv.DictReader(f)
    f_start = []
    f_stop = []
    for row in csv_dict_reader:
        f_start.append(float(row['f_start']))
        f_stop.append(float(row['f_stop']))
META['f_start'] = f_start
META['f_stop'] = f_stop