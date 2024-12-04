import datetime
from collections import defaultdict
from asammdf.blocks import v4_constants as v4c
from asammdf.blocks import utils as asmutls
import numpy as np
# from collections import defaultdict
import pandas as pd
from asammdf import MDF
from J1939_PGN import J1939_PGN, J1939_PDU

import canmatrix.formats
from etils import ecolab
from pathlib import Path
import os
from can_utils.can_dbc_db import CanDbcDb
import sys
import re
from loguru import logger
# import logging
# logger = logging.getLogger(__name__)



def log_explore(dbc_dir,log_file,f_time,map_file,map_ecu_sheet):
    # %%
    file_export=get_report_filename(log_file);
    prepare_report_file(file_export,dbc_dir,log_file,f_time)

    # %%
    dbc_db=CanDbcDb(dbc_dir,export_path=False)
    #%%
    ECU_NAME_MAP=pd.read_excel(map_file,sheet_name=map_ecu_sheet,index_col='node_id_dec')
    ECU_NAME_MAP.head();
    #%%
    mdf1 = MDF(log_file)
    if f_time:
        mdf1=mdf1.cut(f_time[0],f_time[1],whence=1,time_from_zero=1)
    #%%
    mdf1_d,_ =mdf_get_can_msgs(mdf1,get_timestat=True, get_time=True)
    #%% 
    (mdf1_d,mdf1_df)=proc_mdf_all(mdf1,mdf1_d,dbc_db,ecu_name_map=ECU_NAME_MAP)
    mdf1_df.attrs;
    mdf1_df.head() 
    
    #%% stats on dbc
    print_stat_log_msgs(mdf1_df,dbc_db,file_export=file_export)   
    #%%
    get_can_com_matrix(mdf1_df,file_export=file_export)
    
    return (mdf1_d,mdf1_df,dbc_db)


# def calculate_pgn(frame_id):
#     pgn = (frame_id & 0x03FFFF00) >> 8
#     pgn_f = pgn & 0xFF00
#     if pgn_f < 0xF000:
#         pgn &= 0xFFFFFF00
#     return pgn

# def calculate_sa(frame_id):
#     sa = frame_id & 0x000000FF
#     return sa

# def calculate_da(frame_id):
#     da = frame_id & 0x000000FF
#     return da

def mdf_get_trace(mdf):
    """ get CAN trace from mdf file. Output - DataFrame"""
    dfs = []
    logger.info("Processing mdf file to get CAN Trace")

    if mdf.version >= "4.00":

        groups_count = len(mdf.groups)

        for index in range(groups_count):
            group = mdf.groups[index]
            if group.channel_group.flags & v4c.FLAG_CG_BUS_EVENT:
                source = group.channel_group.acq_source

                names = [ch.name for ch in group.channels]

                if source and source.bus_type == v4c.BUS_TYPE_CAN:
                    if "CAN_DataFrame" in names:
                        data = mdf.get("CAN_DataFrame", index)  # , raw=True)

                    elif "CAN_RemoteFrame" in names:
                        data = mdf.get("CAN_RemoteFrame", index, raw=True)

                    elif "CAN_ErrorFrame" in names:
                        data = mdf.get("CAN_ErrorFrame", index, raw=True)

                    else:
                        continue

                    df_index = data.timestamps
                    count = len(df_index)

                    columns = {
                        "timestamps": df_index,
                        "Bus": np.full(count, "Unknown", dtype="O"),
                        "ID": np.full(count, 0xFFFFFFFF, dtype="u4"),
                        "IDE": np.zeros(count, dtype="u1"),
                        "Direction": np.full(count, "", dtype="O"),
                        "Name": np.full(count, "", dtype="O"),
                        "Event Type": np.full(count, "CAN Frame", dtype="O"),
                        "Details": np.full(count, "", dtype="O"),
                        "ESI": np.full(count, "", dtype="O"),
                        "EDL": np.full(count, "Standard CAN", dtype="O"),
                        "BRS": np.full(count, "", dtype="O"),
                        "DLC": np.zeros(count, dtype="u1"),
                        "Data Length": np.zeros(count, dtype="u1"),
                        "Data Bytes": np.full(count, "", dtype="O"),
                        # "pgn": np.full(count, 0, dtype="u2"),
                        # "sa": np.full(count, 0, dtype="u1"),
                        # "da": np.full(count, 0, dtype="u1")
                    }

                    for string in v4c.CAN_ERROR_TYPES.values():
                        sys.intern(string)

                    frame_map = None
                    # if data.attachment and data.attachment[0]:
                    #     dbc = load_can_database(data.attachment[1], data.attachment[0])
                    #     if dbc:
                    #         frame_map = {frame.arbitration_id.id: frame.name for frame in dbc}

                    #         for name in frame_map.values():
                    #             sys.intern(name)

                    if data.name == "CAN_DataFrame":
                        vals = data["CAN_DataFrame.BusChannel"].astype("u1")

                        vals = [f"CAN {chn}" for chn in vals.tolist()]
                        columns["Bus"] = vals

                        vals = data["CAN_DataFrame.ID"].astype("u4") & 0x1FFFFFFF
                        columns["ID"] = vals
                        if frame_map:
                            columns["Name"] = [frame_map.get(_id, "") for _id in vals.tolist()]

                        if "CAN_DataFrame.IDE" in names:
                            columns["IDE"] = data["CAN_DataFrame.IDE"].astype("u1")
                            # j1939 = J1939_PGN(msg_id= columns["ID"])
                            # msg_pgn = j1939.pgn
                            # msg_sa = j1939.sa
                            # if j1939.pdu is J1939_PDU.PDU1:
                            #     # {j1939.ps:02X}
                            #     msg_pdu='PDU1'
                            #     msg_da = j1939.ps
                            # else:
                            #     msg_pdu='PDU2'
                            #     msg_da = None
                                

                            

                        columns["DLC"] = data["CAN_DataFrame.DLC"].astype("u1")
                        data_length = data["CAN_DataFrame.DataLength"].astype("u1")
                        columns["Data Length"] = data_length

                        vals = asmutls.csv_bytearray2hex(
                            pd.Series(list(data["CAN_DataFrame.DataBytes"])),
                            data_length.tolist(),
                        )
                        columns["Data Bytes"] = vals

                        if "CAN_DataFrame.Dir" in names:
                            if data["CAN_DataFrame.Dir"].dtype.kind == "S":
                                columns["Direction"] = [
                                    v.decode("utf-8") for v in data["CAN_DataFrame.Dir"].tolist()
                                ]
                            else:
                                columns["Direction"] = [
                                    "TX" if dir else "RX" for dir in data["CAN_DataFrame.Dir"].astype("u1").tolist()
                                ]

                        if "CAN_DataFrame.ESI" in names:
                            columns["ESI"] = [
                                "Error" if dir else "No error"
                                for dir in data["CAN_DataFrame.ESI"].astype("u1").tolist()
                            ]

                        if "CAN_DataFrame.EDL" in names:
                            columns["EDL"] = [
                                "CAN FD" if dir else "Standard CAN"
                                for dir in data["CAN_DataFrame.EDL"].astype("u1").tolist()
                            ]

                        if "CAN_DataFrame.BRS" in names:
                            columns["BRS"] = [str(dir) for dir in data["CAN_DataFrame.BRS"].astype("u1").tolist()]

                        vals = None
                        data_length = None

                    elif data.name == "CAN_RemoteFrame":
                        vals = data["CAN_RemoteFrame.BusChannel"].astype("u1")
                        vals = [f"CAN {chn}" for chn in vals.tolist()]
                        columns["Bus"] = vals

                        vals = data["CAN_RemoteFrame.ID"].astype("u4") & 0x1FFFFFFF
                        columns["ID"] = vals
                        if frame_map:
                            columns["Name"] = [frame_map.get(_id, "") for _id in vals.tolist()]

                        if "CAN_RemoteFrame.IDE" in names:
                            columns["IDE"] = data["CAN_RemoteFrame.IDE"].astype("u1")

                        columns["DLC"] = data["CAN_RemoteFrame.DLC"].astype("u1")
                        data_length = data["CAN_RemoteFrame.DataLength"].astype("u1")
                        columns["Data Length"] = data_length
                        columns["Event Type"] = "Remote Frame"

                        if "CAN_RemoteFrame.Dir" in names:
                            if data["CAN_RemoteFrame.Dir"].dtype.kind == "S":
                                columns["Direction"] = [
                                    v.decode("utf-8") for v in data["CAN_RemoteFrame.Dir"].tolist()
                                ]
                            else:
                                columns["Direction"] = [
                                    "TX" if dir else "RX"
                                    for dir in data["CAN_RemoteFrame.Dir"].astype("u1").tolist()
                                ]

                        vals = None
                        data_length = None

                    elif data.name == "CAN_ErrorFrame":
                        names = set(data.samples.dtype.names)

                        if "CAN_ErrorFrame.BusChannel" in names:
                            vals = data["CAN_ErrorFrame.BusChannel"].astype("u1")
                            vals = [f"CAN {chn}" for chn in vals.tolist()]
                            columns["Bus"] = vals

                        if "CAN_ErrorFrame.ID" in names:
                            vals = data["CAN_ErrorFrame.ID"].astype("u4") & 0x1FFFFFFF
                            columns["ID"] = vals
                            if frame_map:
                                columns["Name"] = [frame_map.get(_id, "") for _id in vals.tolist()]

                        if "CAN_ErrorFrame.IDE" in names:
                            columns["IDE"] = data["CAN_ErrorFrame.IDE"].astype("u1")

                        if "CAN_ErrorFrame.DLC" in names:
                            columns["DLC"] = data["CAN_ErrorFrame.DLC"].astype("u1")

                        if "CAN_ErrorFrame.DataLength" in names:
                            columns["Data Length"] = data["CAN_ErrorFrame.DataLength"].astype("u1")

                        columns["Event Type"] = "Error Frame"

                        if "CAN_ErrorFrame.ErrorType" in names:
                            vals = data["CAN_ErrorFrame.ErrorType"].astype("u1").tolist()
                            vals = [v4c.CAN_ERROR_TYPES.get(err, "Other error") for err in vals]

                            columns["Details"] = vals

                        if "CAN_ErrorFrame.Dir" in names:
                            if data["CAN_ErrorFrame.Dir"].dtype.kind == "S":
                                columns["Direction"] = [
                                    v.decode("utf-8") for v in data["CAN_ErrorFrame.Dir"].tolist()
                                ]
                            else:
                                columns["Direction"] = [
                                    "TX" if dir else "RX"
                                    for dir in data["CAN_ErrorFrame.Dir"].astype("u1").tolist()
                                ]

                    df = pd.DataFrame(columns, index=df_index)
                    dfs.append(df)

        if not dfs:
            df_index = []
            count = 0

            columns = {
                "timestamps": df_index,
                "Bus": np.full(count, "Unknown", dtype="O"),
                "ID": np.full(count, 0xFFFFFFFF, dtype="u4"),
                "IDE": np.zeros(count, dtype="u1"),
                "Direction": np.full(count, "", dtype="O"),
                "Name": np.full(count, "", dtype="O"),
                "Event Type": np.full(count, "CAN Frame", dtype="O"),
                "Details": np.full(count, "", dtype="O"),
                "ESI": np.full(count, "", dtype="O"),
                "EDL": np.full(count, "Standard CAN", dtype="O"),
                "BRS": np.full(count, "", dtype="O"),
                "DLC": np.zeros(count, dtype="u1"),
                "Data Length": np.zeros(count, dtype="u1"),
                "Data Bytes": np.full(count, "", dtype="O"),
            }
            signals = pd.DataFrame(columns, index=df_index)

        else:
            signals = pd.concat(dfs).sort_index()

            index = pd.Index(range(len(signals)))
            signals.set_index(index, inplace=True)

        del dfs
        
    def strHex2Int(s):
        ss=s.split(' ')
        y=[int(h,16) for h in ss]
        return y
    cols_map={'timestamps':'TimeStamp','Direction':'Dir','Data Length':'DataLength','Data Bytes': 'DataBytes','Bus':'BusChannel'}
    signals.rename(cols_map,inplace=True,axis=1)
    signals['DataBytes']=signals['DataBytes'].apply(strHex2Int)
    signals['BusChannel']=signals['BusChannel'].apply(lambda s: s.replace('CAN ',''))
    
    logger.success("Preparing CAN Trace complete.")
    
    return signals

def decode_dtc(dtc):
    DTC=dict()
    DTC['dtc'] =dtc
    DTC['spn'] = ((dtc & 0xFFFF) | ((dtc >> 5) & 0x70000))
    DTC['fmi'] = ((dtc >> 16) & 0x1F)
    DTC['oc']  = ((dtc >> 24) & 0x7f)
    DTC['cm']  = ((dtc >> 31) & 0x01)
    
    return DTC

def mdf_log_trace_prepare(mdf_trace):
    def strHex2Int(s):
        ss=s.split(' ')
        y=[int(h,16) for h in ss]
        return y
    
    
    # cols_old_name=['timestamps', 'Bus', 'ID', 'IDE', 'Direction', 'Name', 'Event Type',
    #    'Details', 'ESI', 'EDL', 'BRS', 'DLC', 'Data Length', 'Data Bytes']
    # cols_new_name='TimeStamp BusChannel ID IDE DLC DataLength Dir EDL ESI BRS DataBytes'.split(' ')
    # cols_map={'Direction':'Dir','Data Length':'DataLength'}
    cols_map={'timestamps':'TimeStamp','Direction':'Dir','Data Length':'DataLength'}
    mdf_trace.rename(cols_map,inplace=True,axis=1)
    mdf_trace['DataBytes']=mdf_trace['Data Bytes'].apply(strHex2Int)
    mdf_trace['BusChannel']=mdf_trace['Bus'].apply(lambda s: s.replace('CAN ',''))
    
    
    
    


def get_J1939_attr(col_id,id,ide):
    msg_pdu=None
    msg_sa=None
    msg_pgn=None
    msg_da = None
    if ide:
        j1939 = J1939_PGN(msg_id=id)
        msg_pgn = j1939.pgn
        msg_sa = j1939.sa
        if j1939.pdu is J1939_PDU.PDU1:
            # {j1939.ps:02X}
            msg_pdu='PDU1'
            msg_da = j1939.ps
        else:
            msg_pdu='PDU2'
            msg_da = None
    y={col_id:id,'msg_pdu':msg_pdu,'msg_pgn':msg_pgn,'msg_sa':msg_sa,'msg_da':msg_da}
    return y

def can_trace_df_update_j1939_info(df,col_id='ID',col_ide='IDE'):
    """add PGN, PDU, sources and destination adress"""
    
    IDS=np.unique(df[[col_id,col_ide]].values,axis=0)
    ATTR=[get_J1939_attr(col_id,row[0],row[1]) for row in IDS]
    for col in ATTR[0]:
        if col==col_id:
            continue
        df[col]=None
    df_atr=pd.DataFrame(ATTR)
    df_atr.set_index(col_id,inplace=True)
    
    ind_old=df.index.name
    # if default index with no name - drop it
    df=df.reset_index(drop=ind_old==None).set_index(col_id)
    
    df.update(df_atr)
    if ind_old:
        df=df.reset_index().set_index(ind_old)
    else:
         df=df.reset_index()
    
    return df

def mdf_get_can_msgs(mdf, get_payload=False,get_time=False,get_timestat=False):
    """ get from mdf file CAN messages grouped by CAN, ID, extended. Calculate statistics. Return dict[CAN_num,CAN_MSG_ID,IsExtended]"""

    all_can_messages = []
    total_unique_ids = set()
    all_can_msgs=defaultdict()
    for i, group in enumerate(mdf.groups):
        if (
        not group.channel_group.flags & v4c.FLAG_CG_BUS_EVENT
        or group.channel_group.acq_source.bus_type != v4c.BUS_TYPE_CAN
        or 
        not "CAN_DataFrame" in [ch.name for ch in group.channels]
        ):
            continue
        
        mdf._prepare_record(group)
        data = mdf._load_data(group, optimize_read=False)
        for fragment in data:
                        mdf._set_temporary_master(None)
                        mdf._set_temporary_master(mdf.get_master(i, data=fragment))

                        bus_ids = mdf.get(
                            "CAN_DataFrame.BusChannel",
                            group=i,
                            data=fragment,
                        ).samples.astype("<u1")

                        msg_ids = mdf.get("CAN_DataFrame.ID", group=i, data=fragment).astype("<u4")
                        try:
                            msg_ide = mdf.get("CAN_DataFrame.IDE", group=i, data=fragment).samples.astype("<u1")
                        except:
                            msg_ide = (msg_ids & 0x80000000) >> 31

                        msg_ids &= 0x1FFFFFFF

                        data_bytes = mdf.get(
                            "CAN_DataFrame.DataBytes",
                            group=i,
                            data=fragment,
                        ).samples

                        buses = np.unique(bus_ids)
                        
                        for bus in buses:

                            idx_bus = np.argwhere(bus_ids == bus).ravel()
                            bus_t = msg_ids.timestamps[idx_bus]
                            bus_msg_ids = msg_ids.samples[idx_bus]
                            bus_msg_ide = msg_ide[idx_bus]
                            bus_data_bytes = data_bytes[idx_bus]

                            # tmp_pgn = bus_msg_ids >> 8
                            # ps = tmp_pgn & 0xFF
                            # pf = (bus_msg_ids >> 16) & 0xFF
                            # _pgn = tmp_pgn & 0x3FF00
                            # j1939_msg_pgns = np.where(pf >= 240, _pgn + ps, _pgn)
                            # j9193_msg_sa = bus_msg_ids & 0xFF

                            unique_ids = set(zip(bus_msg_ids.tolist(), bus_msg_ide.tolist()))

                            total_unique_ids = total_unique_ids | set(unique_ids)
                            for msg_id, is_extended in sorted(unique_ids):
                                idx_msg = np.argwhere((bus_msg_ids == msg_id) & (bus_msg_ide == is_extended)).ravel()
                                # payload = bus_data_bytes[idx]
                                if is_extended:
                                    # Optional: only if working with J1939
                                    # try to extract J1939 attributes
                                    # TODO: create logic to distinguish CAN extended and J1939
                                    # tmp_pgn = msg_id >> 8
                                    # ps = tmp_pgn & 0xFF
                                    # pf = (msg_id >> 16) & 0xFF
                                    # _pgn = tmp_pgn & 0x3FF00
                                    # msg_pgn = _pgn + ps if pf >= 240 else _pgn
                                    # msg_sa = msg_id & 0xFF
                                    j1939 = J1939_PGN(msg_id=msg_id)
                                    msg_pgn = j1939.pgn
                                    msg_sa = j1939.sa
                                    if j1939.pdu is J1939_PDU.PDU1:
                                        # {j1939.ps:02X}
                                        msg_pdu='PDU1'
                                        msg_da = j1939.ps
                                    else:
                                        msg_pdu='PDU2'
                                        msg_da = None
                                        
                                else:
                                    msg_pdu=None
                                    msg_sa=None
                                    msg_pgn=None
                                    msg_da = None
                                    
                                    
                                if get_payload:
                                    payload = bus_data_bytes[idx_msg]
                                else:
                                    payload=None
                                    

                                if get_time:
                                    msg_t = bus_t[idx_msg]
                                    msg_dt = np.diff(msg_t, prepend=0)
                                    if len(msg_dt) > 1:
                                        msg_dt[0] = msg_dt[1]

                                else:
                                    msg_t=None
                                    msg_dt=None
                                
                                if get_timestat:
                                    t=bus_t[idx_msg]
                                    dt=np.diff(t)
                                    msg_dt_stat=get_stat_vals(dt)
                                else:
                                    msg_dt_stat=None
                                    
                                
                                    
                                out_msg={
                                'CAN_Bus': bus,
                                'msg_id': msg_id,
                                'is_extended': is_extended,
                                'msg_pgn': msg_pgn,
                                'msg_pdu': msg_pdu,  
                                'msg_sa': msg_sa,
                                'msg_da': msg_da,
                                'msg_time': msg_t,
                                'msg_dt': msg_dt,
                                'msg_dt_stat':msg_dt_stat,
                                'msg_data': payload
                                }
                                all_can_messages.append(out_msg)
                                all_can_msgs.update({(bus,msg_id, is_extended):out_msg})
                                

    all_can_msgs_df = pd.DataFrame(all_can_messages)
      
    print('data collected into groups by (CAN,MSG_ID,IsExtended)')
    print(f'number of records: {len(all_can_messages)}')
    
    return all_can_msgs, all_can_msgs_df
    
    
template_stat_dict = {'mean':None,'median':None,'min':None,'max':None,'std':None,'cv':None,'nobs':None,'x':None}

def get_stat_vals(x):
    stat_vals=template_stat_dict.copy()
    if (x is not None):
        if (x.size>0):
            stat_vals={'mean':np.mean(x),'median':np.median(x),'min':np.min(x),'max':np.max(x),
                       'std':np.std(x),'nobs':len(x),'x':x}
            if (abs(stat_vals['mean'])>0):
                stat_vals['cv']=abs(stat_vals['std']/stat_vals['mean'])*100
                # cv - coefficient of variation, also known as normalized root-mean-square deviation (NRMSD), percent RMS, and relative standard deviation (RSD),
                # https://en.wikipedia.org/wiki/Coefficient_of_variation
    return stat_vals

def get_stat_vals_diff(msgs):
    msg1=msgs['msg1']
    msg2=msgs['msg2']
    stat_vals=template_stat_dict.copy()
    
    
    if (msg1['mean'] is not None) & (msg2['mean'] is not None):
        for k in stat_vals:
            if k=='x':
                continue
            stat_vals[k]=msg1[k]-msg2[k]
            
        # stat_vals={'mean':abs(msg1['mean']-msg2['mean']),'min':abs(msg1['min']-msg2['min'])
                #    ,'max':abs(msg1['max']-msg2['max']),'std':abs(msg1['std']-msg2['std']),'nobs':abs(msg1['nobs']-msg2['nobs'])}
        
    return stat_vals
    
def get_mdf_msgs(file,f1_time=None):
    mdf = MDF(file)
    
    if f1_time:
        mdf=mdf.cut(f1_time[0],f1_time[1],whence=1,time_from_zero=1)
    
    all_can_msgs = mdf_get_can_msgs(mdf)
    msg_keys_to=['CAN_Bus', 'msg_id', 'is_extended', 'msg_pgn', 'msg_sa'] #+'msg_dt_stat'
    dt_stat_keys_to=['mean','median', 'min', 'max', 'std','cv', 'nobs']

    msg_d=[]
    for key,msg in all_can_msgs.items():
        # key;
        # msg=all_can_msgs[key]
        dt=np.diff(msg['msg_time'])
        msg_dt_stat=get_stat_vals(dt)
        # msg['msg_dt']=dt
        msg['msg_dt_stat']=msg_dt_stat
        
        # prepare dict with essential info
        row={}
        # row['idx']=key
        
        for msg_k_to in msg_keys_to:
            row[msg_k_to]=msg[msg_k_to]
        
        for dt_stat_k_to in dt_stat_keys_to:
            row['dt_'+dt_stat_k_to]=msg_dt_stat[dt_stat_k_to]
        
        msg_d.append(row)


    mdf_df = pd.DataFrame(msg_d)
    mdf_df.set_index(['CAN_Bus','msg_id','is_extended'],inplace=True,drop=False)
    # mdf_df.set_index(['idx'],inplace=True)
    
    
    mdf_df.attrs["mdf.file_history"]=str(mdf.file_history)
    mdf_df.attrs["mdf.start_time"]=mdf.start_time
    mdf_df.attrs["mdf.name"]=mdf.name
    # mdf_df.attrs;
    # mdf_df.head();
    return (mdf,mdf_df)

def mdf_diff_report(mdf_df1,mdf_df2,file_export_diff:None):
    # another way to do it
    idx1_new=mdf_df1.index.difference(mdf_df2.index)
    idx2_new=mdf_df2.index.difference(mdf_df1.index)
    idx12_common=mdf_df1.index.intersection(mdf_df2.index)

    # combine dataframes
    cols2merge=['CAN_Bus','msg_id','is_extended','msg_pgn','msg_sa']
    # cols_common=['msg_pgn','msg_sa']
    cols2join=['dt_mean','dt_cv','dt_nobs']
    mdf_df1_c=mdf_df1.copy().reset_index(drop=True)
    mdf_df2_c=mdf_df2.copy().reset_index(drop=True)

    # select only common columns and joined
    mdf_df1_c=mdf_df1_c[cols2merge + cols2join]
    mdf_df2_c=mdf_df2_c[cols2merge + cols2join]

    cols_f1={col:f'f1.{col}' for col in cols2join}
    cols_f2={col:f'f2.{col}' for col in cols2join}
    mdf_df1_c.rename(columns=cols_f1,inplace=True)
    mdf_df2_c.rename(columns=cols_f2,inplace=True)
    mdf_df_c=mdf_df1_c.merge(mdf_df2_c,'outer',left_on=cols2merge,right_on=cols2merge,suffixes=('.f1','.f2'),indicator=True)

    # restore index
    mdf_df_c.set_index(['CAN_Bus','msg_id','is_extended'],inplace=True,drop=False)

    # update status  category
    STATUS=['NEW IN F1','NEW IN F2','DT DIFF','EQUAL']
    # mdf_df_c['STATUS']=pd.Categorical(STATUS)
    mdf_df_c['STATUS']=''

    mdf_df_c.loc[idx1_new,'STATUS']='NEW IN F1'
    mdf_df_c.loc[idx2_new,'STATUS']='NEW IN F2'

    # calculate diff

    for col in cols2join:
        mdf_df_c.loc[idx12_common,f'diff.{col}']=round(abs(mdf_df_c.loc[idx12_common,f'f1.{col}']-mdf_df_c.loc[idx12_common,f'f2.{col}'])/mdf_df_c.loc[idx12_common,f'f1.{col}']*100,2)
        

    mdf_df_c.sort_values(by='diff.dt_mean',ascending=False,inplace=True)

    # diff thresholds
    dT_th=5# 5% threshold
    mdf_df_c.loc[mdf_df_c['diff.dt_mean']>dT_th,'STATUS']+='+DT_MEAN DIFF'
    mdf_df_c.loc[mdf_df_c['diff.dt_cv']>dT_th,'STATUS']+='+DT_CV DIFF'
    mdf_df_c.loc[mdf_df_c['diff.dt_nobs']>dT_th,'STATUS']+='+NOBS DIFF'
    # add dbc columns
    mdf_df_c['dbc1']={}
    mdf_df_c['dbc2']={}
    mdf_df_c.loc[mdf_df1.index,'dbc1']=mdf_df1['dbc']
    mdf_df_c.loc[mdf_df2.index,'dbc2']=mdf_df2['dbc']
    

    if file_export_diff:
            with pd.ExcelWriter(file_export_diff,mode='a',if_sheet_exists='replace') as writer:  
                mdf_df_c.to_excel(writer,sheet_name='DIFF_REPORT')
                print(f"report written into {file_export_diff}")
    
    return mdf_df_c

def get_report_filename(log_file):
    (fpath,fname)=os.path.split(log_file)
    (fname,fext) = os.path.splitext(fname)
    fout_path=os.path.join(fpath,fname+'.xlsx');
    return fout_path


def prepare_report_file(file_export,dbc_dir1,log_file1,f_time1,dbc_dir2=None,log_file2=None,f_time2=None):

    with pd.ExcelWriter(file_export,mode='w') as writer: 
        info={'datetime':datetime.datetime.now().__str__(),'dbc_dir1':dbc_dir1,'log_file1':log_file1,
                    'time1':f_time1,'dbc_dir2':dbc_dir2,'log_file2':log_file2,'time2':f_time2 }
        
        info_df=pd.DataFrame.from_dict(info,orient='index')
        
        info_df.to_excel(writer,sheet_name='info')
        print(f'dropped file and written info page into {file_export}')
        
    
def print_stat_log_msgs(mdf_df,dbc_db,file_export=None):
    # stats on dbc
    dbc_empty_idx=[idx for idx,val in mdf_df['dbc'].items() if len(val)==0]
    dbc_used_idx=[idx for idx,val in mdf_df['dbc'].items() if len(val)>0]
    id_dubl= [key for key,val in mdf_df['dbc'].items() if len(val)>1]

    dbc_used=set([dbc_msg['dbc'] for key,val in mdf_df['dbc'].items() if len(val)>0 for dbc_msg in val])
    dbc_msg_used=set([(dbc_msg['dbc'],dbc_msg['msg']) for key,val in mdf_df['dbc'].items() if len(val)>0 for dbc_msg in val])
    dbc_intersec=set(dbc_db.dbc_name_db.keys()).intersection(dbc_used)
    dbc_msgs_db=set([(dbc,msg) for dbc,msgs in dbc_db.dbc_name_db.items() for msg in msgs])
    dbc_msgs_intersec=(dbc_msgs_db).intersection(dbc_msg_used)
    r_txt=[]
    attrs=mdf_df.attrs
    r_txt+=["MDF file attributes:"]
    for atr in attrs:
        r_txt+=[f"\t{atr}: {attrs[atr]}"]
    r_txt+=[f'\nlog msgs total number: {mdf_df.shape[0]} ']
    r_txt+=[f'log number of id wo dbc: {len(dbc_empty_idx)}']
    r_txt+=[f'log number dublicated id: {len(id_dubl)}']
    r_txt+=['']
    r_txt+=[f'dbc_db total number unique dbc { len(dbc_db.dbc_name_db.keys())}']
    r_txt+=[f'log number of used dbc: {len(dbc_used)}']
    r_txt+=['']
    r_txt+=[f'dbc_db total number of messages in db: {len(dbc_msgs_db)}']
    r_txt+=[f'log number of used messages: {len(dbc_msg_used)}']
    # r_txt+=['used msgs:']
    r_txt+=['\nSA with ECU:']
    r_txt+=[set(mdf_df['ecu_name_sa'])]
    r_txt+=['\nSA without ECU']
    r_txt+=[set(mdf_df[(mdf_df['ecu_name_sa'].isna()) & ~(mdf_df['msg_sa'].isna()) ]['msg_sa'])]

    r_txt+=['\n\nmessages with dt stat violation']
    # for idx, row in mdf_df.iterrows()

    # compare mean value with expected
    cols2print=['ecu_name_sa','dbc','dt_mean',  'dt_cv', 
        'dt_stat_mean_diff', 'dt_stat_mean_viol',
        'dt_stat_cv_viol']
    mdf_df_mean_viol=mdf_df.loc[mdf_df['dt_stat_mean_viol']==True].copy()
    mdf_df_cv_viol=mdf_df.loc[mdf_df['dt_stat_cv_viol']==True].copy()
    mdf_df_mean_viol.sort_values(by=['dt_stat_mean_diff'],ascending=False,inplace=True)
    mdf_df_cv_viol.sort_values(by=['dt_cv'],ascending=False,inplace=True)
    

    r_txt+=[f'number of messages with dT mean violation: {mdf_df_mean_viol.shape[0]}']
    for idx,row in mdf_df_mean_viol.head(5).iterrows():
        dbcs=row['dbc']
        dbcs_txt=[dbc['dbc']+"::"+dbc["msg"] for dbc in dbcs]
        dbcs_txt="; ".join(dbcs_txt)
        r_txt+=[f"  {idx}: {row['dt_stat_mean_diff']}%: "+dbcs_txt]
   
    r_txt+=[f'number of messages with CV violation: {mdf_df_cv_viol.shape[0]}']
   
    for idx,row in mdf_df_cv_viol.head(5).iterrows():
        dbcs=row['dbc']
        dbcs_txt=[dbc['dbc']+"::"+dbc["msg"] for dbc in dbcs]
        dbcs_txt="; ".join(dbcs_txt)
        r_txt+=[f"  {idx}: {row['dt_cv']}%: "+dbcs_txt]

    # r_txt+=[mdf_df_mean_viol.loc[:,cols2print].head()]

    # r_txt+=[mdf_df_cv_viol.loc[:,cols2print].head()]
    # print(r_txt)
    print(*r_txt, sep='\n')
    if file_export:
        with pd.ExcelWriter(file_export,mode='a',if_sheet_exists='replace') as writer: 
            mdf_df.to_excel(writer,sheet_name='mdf_df',merge_cells=False)
            r_df=pd.DataFrame(r_txt)
            r_df.to_excel(writer,sheet_name='log_report')
            mdf_df_mean_viol.to_excel(writer,sheet_name='dt_mean_viol')
            mdf_df_cv_viol.to_excel(writer,sheet_name='dt_cv_viol')
            print(f"stat reports written: {file_export}")
             



def proc_mdf_all(mdf,all_can_msgs,dbc_db,ecu_name_map=None,dt_stats_prc_thd={'mean':10,'cv':10}):
    # dt_stats_prc_threshold - thresholds to detect anomalies in time stats
    # mean - diff in % between measured sample mean time vs ecpected in dbc
    # cv - if measured_cv> threshold_cv
    print(dt_stats_prc_thd)
    msg_keys_to=['CAN_Bus', 'msg_id', 'is_extended', 'msg_pgn','msg_pdu','msg_sa','msg_da'] #+'msg_dt_stat'
    dt_stat_keys_to=['mean','median', 'min', 'max', 'std','cv', 'nobs']
    pgn_hex_fun=lambda x: hex(int(x)) if(not pd.isna(x)) else None

    msg_d=[]
    for key,msg in all_can_msgs.items():
        msg_dt_stat=msg['msg_dt_stat']
        # prepare dict with essential info
        row={}
        # row['idx']=key
        for msg_k_to in msg_keys_to:
            row[msg_k_to]=msg[msg_k_to]
        
        for dt_stat_k_to in dt_stat_keys_to:
            row['dt_'+dt_stat_k_to]=msg_dt_stat[dt_stat_k_to]
        # try to find dbc and message
        dbc=None
        if msg['msg_pgn']:
            dbc=dbc_db.dbc_pgn_db[msg['msg_pgn']]
        else:
            dbc=dbc_db.dbc_id_db[msg['msg_id']]
        msg['dbc']=dbc
        row['dbc']=dbc
        ecu_name_sa=None
        ecu_name_da=None
        if ecu_name_map is not None:
            if msg['msg_sa'] in ecu_name_map.index:
                ecu_name_sa=ecu_name_map['node_name'][msg['msg_sa']]
            if msg['msg_da'] in ecu_name_map.index:
                ecu_name_da=ecu_name_map['node_name'][msg['msg_da']]
        
        id_hex=hex(msg['msg_id'])
        pgn_hex=pgn_hex_fun(msg['msg_pgn'])

        
        msg['ecu_name_sa']=ecu_name_sa
        row['ecu_name_sa']=ecu_name_sa
        msg['ecu_name_da']=ecu_name_da
        row['ecu_name_da']=ecu_name_da
        
        msg['id_hex']=id_hex
        row['id_hex']=id_hex
        
        msg['pgn_hex']=pgn_hex
        row['pgn_hex']=pgn_hex
        
        
        if dbc:
            cycle_times=[d['cycle_time']/1000 for d in dbc]
            cy_times=np.array(cycle_times,dtype=float)
            cy_t_max=np.nanmax(cy_times)
            cy_t_min=np.nanmin(cy_times)
            cy_t_rng=[cy_t_min, cy_t_max]
            send_on_ch= [d['SendOnChange'] for d in dbc]
            # se_on_ch=np.array(send_on_ch,dtype=float)
            
        else:
            cy_t_max=None
            cy_t_min=None
            send_on_ch=None
            cy_t_rng=None
            
        
        msg['cycle_time_range'] = cy_t_rng
        row['cycle_time_range'] = cy_t_rng
        # msg['cycle_time_min'] = cy_t_min
        # row['cycle_time_min'] = cy_t_min
        # msg['cycle_time_max'] = cy_t_max
        # row['cycle_time_max'] = cy_t_max
        row['SendOnChange'] = send_on_ch
        msg['SendOnChange'] = send_on_ch
                
        if row['msg_id']==33423707:
            x=1
        
        # check time stat
        dt_stat_cv_viol=None
        dt_stat_mean_viol=None
        dt_diff = None
        if msg_dt_stat['cv']:
            dt_stat_cv_viol=abs(msg_dt_stat['cv'])>dt_stats_prc_thd['cv']
        # this violation could be checked only vs dbc
        if msg_dt_stat['mean'] and dbc:
            dt_m_thd=dt_stats_prc_thd['mean']
            dt_m=msg_dt_stat['mean']
            for dbc_item in dbc:
                dt_exp=dbc_item['cycle_time']/1000 # cycle time is given in ms
                if abs(dt_exp)<0.001: 
                    # for case of zero time take the min. possible measured
                    # continue - we can't estimate
                    continue
                    dt_exp=0.001
                dt_diff=abs(dt_m-dt_exp)/dt_exp*100
                if dt_diff>dt_m_thd:
                    dt_stat_mean_viol=True
                    break
                
        row['dt_stat_mean_diff']=dt_diff
        row['dt_stat_mean_viol']=dt_stat_mean_viol
        row['dt_stat_cv_viol']=dt_stat_cv_viol
        
        
        
        msg_d.append(row)


    mdf_df = pd.DataFrame(msg_d)
    mdf_df.set_index(['CAN_Bus','msg_id','is_extended'],inplace=True,drop=False)


    #  add meta info
    mdf_df.attrs["mdf.file_history"]=str(mdf.file_history)
    mdf_df.attrs["mdf.start_time"]=mdf.start_time
    mdf_df.attrs["mdf.name"]=mdf.name
    mdf_df.sort_index(inplace=True,sort_remaining=True)
    
    return (all_can_msgs,mdf_df)
    
def get_can_com_matrix(mdf_df,file_export=None):
    # prepare can matrix
    d=defaultdict(lambda:  defaultdict())
    node_c_da=defaultdict(lambda:  defaultdict())
    node_c=defaultdict(lambda:  defaultdict())
    dm=defaultdict(lambda:  defaultdict(lambda:  defaultdict()))
    for idx,row in mdf_df.iterrows():
        CAN_Bus=idx[0]
        if not pd.isna(row['msg_sa']):
            if row['ecu_name_sa']:
                ecu_name_sa=row['ecu_name_sa']
            else:
                ecu_name_sa='NA'
            if row['ecu_name_da']:
                ecu_name_da=row['ecu_name_da']
            else:
                ecu_name_da='NA'
            if not pd.isna(row['msg_da']):
                msg_da=str(row['msg_da'])
            else:
                msg_da='ALL'
                
            node_c[(row['msg_sa'],ecu_name_sa)][f'CAN{row['CAN_Bus']}']=1
            # node_c[(row['msg_sa'],ecu_name_sa)][CAN_Bus]=1
            node_c_da[(row['msg_sa'],ecu_name_sa)][(f'CAN{row['CAN_Bus']}',msg_da,ecu_name_da)]=1
            # node_c_da[(row['msg_sa'],ecu_name_sa)][CAN_Bus,msg_da,ecu_name_da)]=1

        if len(row['dbc'])>0:
            for dbc_item in row['dbc']:
                # d[dbc_item['dbc']][f'{idx[0]}']=1
                d[dbc_item['dbc']][f'CAN{row['CAN_Bus']}']=1
                dm[(dbc_item['dbc'],dbc_item['msg'])][f'CAN{row['CAN_Bus']}']=1
                # dm[(dbc_item['dbc'],dbc_item['msg'])][CAN_Bus]=1
                
    CAN_DBC_MAT=pd.DataFrame.from_dict(d,orient='index')
    CAN_DBC_MSG_MAT=pd.DataFrame.from_dict(dm,orient='index')
    CAN_NODE_MAT=pd.DataFrame.from_dict(node_c,orient='index')
    CAN_NODE_DA_MAT=pd.DataFrame.from_dict(node_c_da,orient='index')
    # CAN_DBC_MAT.fillna(value=False,inplace=True)

    # CAN_DBC_MAT.head()

    # CAN_DBC_MSG_MAT.index.set_names(['dbc','msg'])
    # CAN_DBC_MSG_MAT.fillna(value=False,inplace=True)
    CAN_DBC_MAT.sort_index(inplace=True)
    CAN_DBC_MAT.sort_index(axis=1,sort_remaining=True, inplace=True)
    CAN_DBC_MSG_MAT.sort_index(inplace=True)
    CAN_DBC_MSG_MAT.sort_index(axis=1,sort_remaining=True,inplace=True)
    CAN_NODE_MAT.sort_index(inplace=True)
    CAN_NODE_MAT.sort_index(axis=1,sort_remaining=True, inplace=True)
    CAN_NODE_DA_MAT.sort_index(inplace=True)
    CAN_NODE_DA_MAT.sort_index(axis=1, sort_remaining=True, inplace=True)
    # CAN_DBC_MSG_MAT.head()
    if file_export:
        with pd.ExcelWriter(file_export,mode='a',if_sheet_exists='replace') as writer:  
            CAN_NODE_MAT.to_excel(writer,sheet_name='CAN_NODES')
            CAN_NODE_DA_MAT.to_excel(writer,sheet_name='CAN_NODES_DA')
            CAN_DBC_MAT.to_excel(writer,sheet_name='CAN_DBC')
            CAN_DBC_MSG_MAT.to_excel(writer,sheet_name='CAN_DBC_MSG') 
            print(f"CAN MATRIX reports written: {file_export}")
    
    return (CAN_DBC_MAT,CAN_DBC_MSG_MAT,CAN_NODE_MAT,CAN_NODE_DA_MAT)
    


def get_can_map2paths(dbc_dir, CAN_MAP,CH_NAME):
    dbc_files=list(Path(dbc_dir).glob("*.dbc"))
    CAN_SEL=CAN_MAP[CAN_MAP[CH_NAME]][['dbc',CH_NAME]]

    filtered_files = CAN_SEL['dbc'].tolist()

    # Get a list of file names from the files list for easy lookup
    available_file_names = {file.name: file for file in dbc_files}

    # Initialize a list for matching paths
    CAN_PTHS = []

    # Iterate over filtered_files and check if each file is in available_file_names
    for dbc_file in filtered_files:
        if dbc_file in available_file_names:
            # Add the matching path if found
            CAN_PTHS.append(available_file_names[dbc_file])
        else:
            # Print a warning if the file is not found in the list
            print(f"Warning: {dbc_file} not found in the available files list.")
    # Print the matching paths
    # print("Matching paths:")
    # print(CAN_PTHS)
    return CAN_PTHS


def can_map_proc(can_map,ch_col_start=1):
    """process CAN MAP df table: cols[1:].astype boolean and reduce it"""
    can_map.fillna(value=0,inplace=True)
    ch_map_cols=can_map.columns[ch_col_start:]
    can_map[ch_map_cols]=can_map[ch_map_cols].astype('bool')
    can_ch=can_map[ch_map_cols]
    can_map1=can_map.drop(ch_map_cols,axis=1)
    idx1=can_ch.any(axis=1,bool_only=True) # rows
    idx2=can_ch.any(axis=0) # cols
    can_map1 = can_map1.loc[idx1,:]
    can_ch = can_ch.loc[idx1,:]
    can_ch=can_ch.loc[:,idx2]
    can_map2=pd.concat([can_map1,can_ch],axis=1)
    can_map2.reset_index(inplace=True,drop=True)
    
    ch_map_cols=list(ch_map_cols.values)
    
    return can_map2, ch_map_cols
    
    
    
    
def get_can_map(file,sheet=0,ch_col_start=1):
    can_map=pd.read_excel(file,sheet_name=sheet)
    can_map,can_map_ch=can_map_proc(can_map,ch_col_start)
    return can_map, can_map_ch

def get_can_map_db(dbc_dir,can2ch_maps,can_map_df):
    can_db=[]
    for can_name in can2ch_maps:
        CAN_CH=get_can_map2paths(dbc_dir,can_map_df,can_name)
        can_db1=[(path,can2ch_maps[can_name]) for path in CAN_CH]
        can_db.extend(can_db1)
    
    can_db = {"CAN": can_db}
    return can_db

def get_can_map2extract(dbc_dir,can2ch_maps,file,sheet=0):

    can_map_df,_=get_can_map(file,sheet)
    can_db = get_can_map_db(dbc_dir,can2ch_maps,can_map_df)
    return (can_map_df,can_db)

def df_decode_bytes(df):
    for col in df.columns:
        if df[col].dtype == object and df[col].dropna().apply(lambda x: isinstance(x, bytes)).all():
            df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
    return df
        
def mdf_extracted_to_dict_df(mdf1_extracted):
    
    mdf1_msgs=defaultdict(lambda: defaultdict(list))
    mdf1_msgs_grp=defaultdict(lambda: defaultdict(int))
    mdf1_can_sa_msgs_grp=defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    mdf1_can_sa_msgs=defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    mdf1_can_sa_msgs_sigs_idx=defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(tuple))))
    #                               can             sa                  msg                 sig

    mdf1_all_signames=[list(can_signal.display_names.keys())[0] for can_signal in mdf1_extracted]

    for can_signal in mdf1_extracted:
        can_msg_sig=list(can_signal.display_names.keys())[0]
        msg=can_msg_sig.split('.')
        m_can=msg[0]
        m_msg=msg[1]
        m_sig=msg[2]
        src=can_signal.source.name
        src_sa='NO'
        m_gr_ch_idx=(can_signal.group_index, can_signal.channel_index)
        if 'SourceAddress' in src:
            # get SA address of the sender
            if '0x' in src:
                m=src.split('0x')
                if len(m)>1:
                    src_sa=m[1]
                else:
                    src_sa=src
            else:
                    src_sa=src
            m=mdf1_can_sa_msgs

        else:
            src_sa='0x0'
        if msg[2] not in  mdf1_can_sa_msgs[msg[0]][src_sa][msg[1]]:
            mdf1_can_sa_msgs[msg[0]][src_sa][msg[1]].append(msg[2])
            
        if m_sig not in  mdf1_can_sa_msgs_sigs_idx[m_can][src_sa][m_msg]:
            mdf1_can_sa_msgs_sigs_idx[m_can][src_sa][m_msg][m_sig]=m_gr_ch_idx
            
        if msg[2] not in mdf1_msgs[msg[0]][msg[1]]:
            mdf1_msgs[msg[0]][msg[1]].append(msg[2])
        # detect multiple groups
        grp_sig_id=mdf1_extracted.channels_db[can_msg_sig]
        grp_sig_id_sum=sum(len(x) for x in grp_sig_id)
        if grp_sig_id_sum>2:
            print(f"group already exist for {can_signal.display_names} {grp_sig_id}. Take the first one")

        mdf1_msgs_grp[msg[0]][msg[1]]=grp_sig_id[0][0]
        mdf1_can_sa_msgs_grp[msg[0]][src_sa][msg[1]]=grp_sig_id[0][0]
   

    # mdf1_msgs_grp;
    # mdf1_msgs;
    msgs1_df=defaultdict(lambda: defaultdict(list))
    for can,can_msgs in mdf1_msgs_grp.items():
        
        for msg, grp in can_msgs.items():
            df=mdf1_extracted.get_group(grp,use_display_names=False)
            df=df_decode_bytes(df)
            msgs1_df[msg]=df
            
    return mdf1_msgs_grp,msgs1_df


 
    
    
    
    
        