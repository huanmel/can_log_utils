# %%
import pandas as pd
import numpy as np
from asammdf import MDF, Signal
from asammdf.blocks.source_utils import Source
import os
import asammdf.blocks.v4_constants as v4c

def convert_log_csv2raw_can_mdf(csv_file):
    # %%
    # Read the CSV file

    mdf_file= os.path.abspath(csv_file).replace('.csv','.mf4')

    df = pd.read_csv(csv_file,parse_dates=['Time'])
    df['Time'] = pd.to_datetime(df['Time']).dt.tz_localize('UTC')
  # %%
    # Define the data type structure
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

    # %%
    # Clean and prepare data
    df['ID'] = df['ID'].astype(np.uint32)  # Ensure ID is uint32
    df['Extended'] = df['Extended'].astype(np.uint8)  # Ensure IDE is uint8
    df['Length'] = df['Length'].astype(np.uint8)  # Ensure DLC and DataLength are uint8


    # %%
    # Prepare arrays for the structured array

    bus_channel = df['BusChannel'].values.astype(dtype='u1')  # Empty BusChannel (could use 'Name' if populated)

    id_array = np.where(
        df['Extended'],
        df['ID'] | (1 << 31),  # Set IDE bit for extended frames
        df['ID']               # Leave standard frames unchanged
    ).astype(dtype='u4')

    ide_array = df['Extended'].values.astype(dtype='u1')
    dlc_array = df['Length'].values.astype(dtype='u1')
    data_length_array = df['Length'].values.astype(dtype='u1')
    data_bytes_array = np.zeros((len(df), 8), dtype=np.uint8)
    data_bytes_array[:, :8] = df[['Data_1', 'Data_2', 'Data_3', 'Data_4', 'Data_5', 'Data_6', 'Data_7', 'Data_8']].values
    dir_array = np.where(df['Dir'] == 1, 1, 0).astype(np.uint8)  # 0 = received, 1 = transmitted
    edl_array = np.zeros(len(df), dtype=np.uint8)  # Default to 0 for non-CAN FD
    brs_array = np.zeros(len(df), dtype=np.uint8)  # Default to 0 for non-CAN FD
    esi_array = np.zeros(len(df), dtype=np.uint8)  # Default to 0 for non-CAN FD



    # %%
    # Create structured array
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

    # Define source information
    source = Source(
        name="",
        path="CAN_DEVICE",
        comment="",
        source_type=v4c.SOURCE_BUS,
        bus_type=v4c.BUS_TYPE_CAN,
    )
    t=df['Time']
    starttime=t[0]
    time=t-t[0]
    timestamps=time.dt.total_seconds()


    # Create Signal object
    sig = Signal(
        samples=structured_array,
        timestamps=timestamps,  # Use a range for timestamps
        # timestamps=df['Time'].values,
        name='CAN_DataFrame',
        source=source,
    )
    # Create and save MDF file
    mdf = MDF()


    mdf.start_time=starttime
    mdf.append(sig)
    mdf.save(mdf_file, overwrite=True)
    print(f'log file converted to {mdf_file}')




