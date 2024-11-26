import pandas as pd
from pyproj import Proj, transform              # for coordinate transformations

#read the following csv file: sensitivity_analysis/outputs/forest-food-percent/forest-food-percent:0.1/prob_water__0.0__output_files/food_value_forest__5__food_value_cropland__100/THRESHOLD_32/solitary_bulls/aggression:0.2/model_01/Aug/627e6251-235e-47c6-a65c-07db782cfa49/output_files/agent_data.csv
data = pd.read_csv('sensitivity_analysis/outputs/forest-food-percent/forest-food-percent:0.1/prob_water__0.0__output_files/food_value_forest__5__food_value_cropland__100/THRESHOLD_32/solitary_bulls/aggression:0.2/model_01/Aug/627e6251-235e-47c6-a65c-07db782cfa49/output_files/agent_data.csv')

elephant_data = data[data['AgentID'] == 'bull_0']

#select columns: Step: longitude, latritude
elephant_data = elephant_data[['Step', 'longitude', 'latitude']]

#convert lat, lon from epsg:3857 to epsg:4326

inProj = Proj(init='epsg:3857')
outProj = Proj(init='epsg:4326')
elephant_data['longitude'], elephant_data['latitude'] = transform(inProj, outProj, elephant_data['longitude'], elephant_data['latitude'])

#starting from zeroth Step, create time stamp for each row at 5 minutes interval
elephant_data['timestamp'] = elephant_data['Step'].apply(lambda x: x*5)

start_date = '2010-01-01 00:00:00'
start_date = pd.to_datetime(start_date)
elephant_data['timestamp'] = start_date + pd.to_timedelta(elephant_data['timestamp'], unit='m')

elephant_data['t'] = pd.to_datetime(elephant_data['timestamp'])

print(elephant_data)

#save as csv
elephant_data.to_csv('sensitivity_analysis/plots/preprocess_for_space_use_calculation_AKDEC/elephant_data.csv', index=False)

