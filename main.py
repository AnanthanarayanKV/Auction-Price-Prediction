from cleaning import dataCleaning
from pathlib import Path


cwd = Path.cwd()
path = Path(f"{cwd}/cleaned_data.csv")

if path.is_file():
    print("Cleaned data available and ready to use")
    
else:
    print("Cleaned data not found,\ncleaning data....\n")
    dataCleaning()
    
