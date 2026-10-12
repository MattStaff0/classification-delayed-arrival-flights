import pandas as pd


def main(): 
    df = pd.read_csv("data/raw/2025-07-flight-data.csv", low_memory=False)
    blks = df[["DepTimeBlk", "ArrTimeBlk"]]

    # Unique block values
    dep_block_vals = blks["DepTimeBlk"].value_counts().sort_index(ascending=True)
    arr_block_vals = blks["ArrTimeBlk"].value_counts().sort_index(ascending=True)

    print(type(dep_block_vals))

    # Prints
    print(f'Depature Block Values:\n{dep_block_vals}')
    print(f'Arrival Block Values:\n{arr_block_vals}')


    

main()