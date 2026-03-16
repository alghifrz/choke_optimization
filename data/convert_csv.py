import pandas as pd
df = pd.read_excel('Volve_production_data.xlsx', sheet_name='Daily Production Data')
df.to_csv('volve_production_data.csv', index=False)