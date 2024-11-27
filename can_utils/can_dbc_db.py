from collections import defaultdict
from pathlib import Path
import pandas as pd
import canmatrix.formats
import os
import decimal

class CanDbcDb:
    SIG_ATTR=['name','unit','comment','initial_value','min','max','factor','enumeration','values']
    
    def __init__(self,dbc_dir,print_id_conlficts=False,export_path=None) -> None:
        dbc_files=Path(dbc_dir).glob("*.dbc")
        dbc_files_list=list(dbc_files)
        dbc_name_db=defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        
        dbc_id_db=defaultdict(list)
        dbc_pgn_db=defaultdict(list)
        for dbc_file in dbc_files_list:
            dbc = canmatrix.formats.loadp_flat(str(dbc_file))
            dbc_name=os.path.basename(dbc_file)
            for frame in dbc.frames:
                # sigs=[sig.name for sig in frame.signals]
                # sigs=[]
                sig_names=[sig.name for sig in frame.signals]
                sigs=defaultdict(dict)
                for sig in frame.signals:
                    # sig_d={k:v for k, v in sig.__dict__.items() if k in DbcDb.SIG_ATTR}
                    sig_d={key: sig.__dict__[key] for key in DbcDb.SIG_ATTR if key in sig.__dict__}
                    for k,v in sig_d.items():
                        if isinstance(v,decimal.Decimal):
                            sig_d[k]=float(v)
                    sigs[sig.name]=sig_d
                    
                info=defaultdict(None)
                info={'id':frame.arbitration_id.id,
                    'is_extended':frame.arbitration_id.extended,
                    'is_j1939':frame.is_j1939,
                    'comment':frame.comment,
                    'cycle_time':frame.cycle_time,
                    'attributes': frame.attributes           
                }
                #  prepare entry for _db dicts
                db_entry={'dbc':dbc_name,'msg':frame.name,
                    'is_j1939':frame.is_j1939,'cycle_time':frame.cycle_time}
                if 'SendOnChange' in frame.attributes:
                    db_entry['SendOnChange']=frame.attributes['SendOnChange']
                else:
                    db_entry['SendOnChange']=None
                    
                if frame.is_j1939:
                    info['pgn']=frame.pgn
                    info['SA']=frame.arbitration_id.j1939_source
                    info['DA']=frame.arbitration_id.j1939_destination
                    dbc_pgn_db[frame.pgn].append(db_entry)
                    
                    # dbc_id_db[frame.pgn].append({'dbc':dbc_name,'msg':frame.name,
                    # 'is_j1939':frame.is_j1939})
                    
                # else:# CAN
                dbc_id_db[frame.arbitration_id.id].append(db_entry)
                    
                    
                dbc_name_db[dbc_name][frame.name]['info']=info
                dbc_name_db[dbc_name][frame.name]['signals']=sigs
                # dbc_name_db[dbc_name][frame.name]['signals']=sigs
                # dbc_name_db[dbc_name][frame.name]['signals']=sigs
        
        id_dubl= [key for key,val in dbc_id_db.items() if len(val)>1]
        pgn_dubl= [key for key,val in dbc_pgn_db.items() if len(val)>1]
        dbc_msgs_db=set([(dbc,msg) for dbc,msgs in dbc_name_db.items() for msg in msgs])
        
        print(f'total number of id: {len(dbc_id_db.keys())}')
        print(f'total number of dbc.msg: {len(dbc_msgs_db)}')
        print(f'number of dublicated id: {len(id_dubl)}')
        print(f'number of dublicated pgn: {len(pgn_dubl)}')
        
        # update db with dublicated info
        
        for id in set(id_dubl):
            dbcs=dbc_id_db[id]
            dbcs_list=[dbc['dbc']+'::'+dbc['msg'] for dbc in dbcs]
            for dbc in dbcs:
                dbc_name_db[dbc['dbc']][dbc['msg']]['info']['dbc_dubl']=dbcs_list
        
        self.dbc_name_db=dbc_name_db
        self.dbc_id_db=dbc_id_db
        self.dbc_pgn_db=dbc_pgn_db
        self.id_dubl=id_dubl
        
        # todo messages with different names but dublicated id
        if print_id_conlficts:
            self.print_id_dubl()
        # if export_path:
        self.get_dbc_db_df(export_path)
        print('dbc db loaded')
            
    def print_id_dubl(self,print_all=False):
        print('possible id conflicts:')
        print('\tone id but different message name')
        for id in self.id_dubl:
            if print_all:
                msgs=[item['dbc']+':'+item['msg'] for item in self.dbc_id_db[id]]
                
                print(f"\t\tid={id} msg names:{msgs}")
                # print(self.dbc_id_db[id])
            else:
                msgs=[item['msg'] for item in self.dbc_id_db[id]]
                
                if len(set(msgs))>1:
                    print(f"\t\tid={id} msg names: {set(msgs)}")
    
    
    def get_dbc_db_df(self,export_path=None):
    # convert to dataframe and save

        df=pd.DataFrame()
        # L=[]
        for dbc_name,dbc_item in self.dbc_name_db.items():
            for msg_name,msg_item in dbc_item.items():
                sigs=msg_item['signals']
                msg_item2=msg_item.copy()
                msg_item2.pop("signals", None)
                row=pd.json_normalize(msg_item2)
                # row.drop('signals')
                row['dbc']=dbc_name
                row['msg']=msg_name
                # row['signals']=[[sig['name'] for sig in msg_item['signals']]]
                row['signals']=[list(sigs.keys())]
                df=pd.concat([df,row])
        # reorder
        df.columns=df.columns.str.replace('info.','')
        df.columns=df.columns.str.replace('attributes.','attr.')
        
        # signals=
        dbc=df.pop('dbc')
        msg=df.pop('msg')
        id=df.pop('id')
        id_hex=id.apply(hex)
        pgn=df.pop('pgn')
        pgn_hex=pgn.apply(lambda x: hex(int(x)) if(not pd.isna(x)) else None)

        df.insert(0,'dbc',dbc)
        df.insert(1,'msg',msg)
        df.insert(2,'id',id)
        df.insert(3,'id_hex',id_hex)
        df.insert(4,'pgn',pgn)
        df.insert(5,'pgn_hex',pgn_hex)
        df.rename({'sig_names':'signals'},inplace=True)
        
        # df.to_excel('dbc_db.xlsx')
        # df.head()
        df.reset_index(inplace=True)
        self.get_dbc_msg_sig_df()
        if export_path:
            folder_name=os.path.split(export_path)[-1]
            export_report=os.path.join(export_path,folder_name+'_db.xlsx')
            df.to_excel(export_report)
            print(f'dbc_msg db saved to {export_report}')
            export_report1=os.path.join(export_path,folder_name+'msg_sig_db.xlsx')
            self.df_sig.to_excel(export_report1)
            print(f'dbc msg sig saved to {export_report1}')
            
            
        self.df=df
        
        return df
    
    def get_dbc_msg_sig_df(self):
        L=[]
        for dbc_name,dbc in self.dbc_name_db.items():
            for msg_name,msg in dbc.items():
                for sig_name,sig in msg['signals'].items():
                    row={'dbc':dbc_name,'msg':msg_name,'signal':sig_name}
                    sig1=sig.copy()
                    sig1.pop('name')
                    row.update(sig1)
                    L.append(row)
                    
        df=pd.DataFrame(L)
        self.df_sig=df
        return df