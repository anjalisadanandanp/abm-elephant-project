#given an .csv files, sample every nth row and create a new csv file

import pandas as pd

df = pd.read_csv('elephant_data.csv')

n=10
df = df.iloc[::n]

df.to_csv('sensitivity_analysis/plots/preprocess_for_space_use_calculation_AKDEC/elephant_data_v1.1.csv', index=False)
