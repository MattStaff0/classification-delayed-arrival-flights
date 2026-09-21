"""
Python Script that has utility functions for the project proposal
"""

from pathlib import Path
import pandas as pd
import os

MAPPING = {
    0: "July 2025",
    1: "August 2025",
    2: "September 2025",
    3: "October 2025",
    4: "November 2025",
    5: "December 2025",
    6: "January 2026",
    7: "February 2026",
    8: "March 2026",
    9: "April 2026",
    10: "May 2026",
    11: "June 2026",
}

COLS = ["FlightDate","Month","DayofMonth","DayOfWeek","Reporting_Airline",
        "Origin","Dest","CRSDepTime","CRSArrTime","CRSElapsedTime",
        "Distance","ArrDel15","Cancelled","Diverted"]


def memory_check(path):
    df = pd.read_csv(path, usecols=COLS)
    mb = df.memory_usage(deep=True).sum() / 1e6

    print(f"File on disk: {os.path.getsize(path)/1e6:.0f} MB")
    print(f"Rows: {len(df):,}")
    print(f"In memory (14 cols): {mb:.0f} MB")
    print(f"Estimated full year: {mb*12/1000:.2f} GB")


def find_raw_instances():
    total_flights_per_month = []
    total_flights = 0
    for f in sorted(Path("data/raw").glob("*-flight-data.csv")):
        n = len(pd.read_csv(f, usecols=["Year"]))
        total_flights_per_month.append(n)
        total_flights += n

    return (total_flights_per_month, total_flights)

def find_usable_instances():
    monthly_total_usable_instances = []
    monthly_total_usable_late_instances = []
    total_usable_instances = 0
    total_usable_late_instances = 0

    for f in sorted(Path("data/raw").glob("*-flight-data.csv")):
        df = pd.read_csv(f, usecols=["Cancelled", "Diverted", "ArrDel15"])
        usable = df[(df.Cancelled == 0) & (df.Diverted == 0) & df.ArrDel15.notna()]
        n_usable = len(usable)
        n_late = int(usable.ArrDel15.sum())
        monthly_total_usable_instances.append(n_usable)
        monthly_total_usable_late_instances.append(n_late)
        total_usable_instances += n_usable
        total_usable_late_instances += n_late

    return (monthly_total_usable_instances, monthly_total_usable_late_instances,
            total_usable_instances, total_usable_late_instances)

def report_totals():
    total_flights_per_month, total_flights = find_raw_instances()
    (monthly_total_usable_instances, monthly_total_usable_late_instances,
     total_usable_instances, total_usable_late_instances) = find_usable_instances()

    print(f'{"Month":<16}{"Total Flights":>15}{"Usable":>15}{"Usable Late":>15}{"Late %":>10}')
    print("-" * 71)

    for index, month in MAPPING.items():
        print(f'{month:<16}{total_flights_per_month[index]:>15,}'
              f'{monthly_total_usable_instances[index]:>15,}'
              f'{monthly_total_usable_late_instances[index]:>15,}'
              f'{monthly_total_usable_late_instances[index] / monthly_total_usable_instances[index]:>10.1%}')

    print("-" * 71)
    print(f'{"TOTAL":<16}{total_flights:>15,}'
          f'{total_usable_instances:>15,}'
          f'{total_usable_late_instances:>15,}'
          f'{total_usable_late_instances / total_usable_instances:>10.1%}')


def main():
    report_totals()
    memory_check(path="data/raw/2025-07-flight-data.csv")


if __name__ == "__main__":
    main()