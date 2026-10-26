import pandas as pd
from modules.exploratory_analysis import service_volume,breach_by_office,breach_by_queue,eligible_labels
def dashboard_summary(df):
    labels=eligible_labels(df); breach=float(labels['SLA_Breached'].astype(int).mean()*100) if len(labels) else None
    return {'total_requests':int(len(df)),'eligible_completed':int(len(labels)),'breach_rate_percent':breach,'pending_requests':int(df['Status'].eq('Pending').sum()) if 'Status' in df else 0,'high_queue_requests':int(df['Queue_Category'].astype('string').eq('High').sum()) if 'Queue_Category' in df else 0}
def flagged_queue(df):
    cols=[c for c in ['Request_ID','Service_Type','Assigned_Office','Priority','Queue_Category','Status','SLA_Breached'] if c in df]
    d=df.loc[df.get('Queue_Category',pd.Series(index=df.index,dtype='object')).astype('string').eq('High')].copy()
    if 'Status' in d: d=d.loc[d['Status'].eq('Pending')]
    return d[cols].head(100)
