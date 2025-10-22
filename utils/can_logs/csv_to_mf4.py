import pandas as pd
import numpy as np
from asammdf import MDF, Signal
from asammdf.blocks.source_utils import Source
import os
import asammdf.blocks.v4_constants as v4c
import argparse

def convert_log_csv2raw_can_mdf(csv_file):
    mdf_file = os.path.abspath(csv_file).replace('.csv','.mf4')

    df = pd.read_csv(csv_file, parse_dates=['Time'])
    df['Time'] = pd.to_datetime(df['Time']).dt.tz_localize('UTC')

    DTYPE = np.dtype(
        [
            ("CAN_DataFrame.BusChannel", "<u1"),
            ("CAN_DataFrame.ID", "<u4"),
            ("CAN_DataFrame.IDE", "<u1"),
            ("CAN_DataFrame.DLC", "<u1"),
            ("CAN_DataFrame.DataLength", "<u1"),
            ("CAN_DataFrame.DataBytes", "(8,)u1"),
            ("CAN_DataFrame.Dir", "<u1"),
            ("CAN_DataFrame.EDL", "<u1"),
            ("CAN_DataFrame.BRS", "<u1"),
            ("CAN_DataFrame.ESI", "<u1"),
        ]
    )

    df['ID'] = df['ID'].astype(np.uint32)
    df['Extended'] = df['Extended'].astype(np.uint8)
    df['Length'] = df['Length'].astype(np.uint8)

    bus_channel = df['BusChannel'].values.astype(dtype='u1')
    id_array = np.where(
        df['Extended'],
        df['ID'] | (1 << 31),
        df['ID']
    ).astype(dtype='u4')
    ide_array = df['Extended'].values.astype(dtype='u1')
    dlc_array = df['Length'].values.astype(dtype='u1')
    data_length_array = df['Length'].values.astype(dtype='u1')
    data_bytes_array = np.zeros((len(df), 8), dtype=np.uint8)
    data_bytes_array[:, :8] = df[['Data_1', 'Data_2', 'Data_3', 'Data_4', 'Data_5', 'Data_6', 'Data_7', 'Data_8']].values
    dir_array = np.where(df['Dir'] == 1, 1, 0).astype(np.uint8)
    edl_array = np.zeros(len(df), dtype=np.uint8)
    brs_array = np.zeros(len(df), dtype=np.uint8)
    esi_array = np.zeros(len(df), dtype=np.uint8)

    arrays = [
        bus_channel,
        id_array,
        ide_array,
        dlc_array,
        data_length_array,
        data_bytes_array,
        dir_array,
        edl_array,
        brs_array,
        esi_array,
    ]

    structured_array = np.core.records.fromarrays(arrays, dtype=DTYPE)

    source = Source(
        name="",
        path="CAN_DEVICE",
        comment="",
        source_type=v4c.SOURCE_BUS,
        bus_type=v4c.BUS_TYPE_CAN,
    )
    t = df['Time']
    starttime = t[0]
    time = t - t[0]
    timestamps = time.dt.total_seconds()

    sig = Signal(
        samples=structured_array,
        timestamps=timestamps,
        name='CAN_DataFrame',
        source=source,
    )
    
    mdf = MDF()
    mdf.start_time = starttime
    mdf.append(sig)
    mdf.save(mdf_file, overwrite=True)
    print(f'Log file converted to {mdf_file}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert CSV log file to CAN MDF (.mf4) format')
    parser.add_argument('csv_file', help='Path to the input CSV file')
    args = parser.parse_args()
    
    convert_log_csv2raw_can_mdf(args.csv_file)