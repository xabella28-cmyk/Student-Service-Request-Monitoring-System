import pandas as pd
import plotly.express as px
def service_volume(df): return df['Service_Type'].fillna('(Missing)').value_counts().rename_axis('Service_Type').reset_index(name='Request_Count')
def eligible_labels(df):
    if 'SLA_Breached' not in df: return df.iloc[0:0].copy()
    return df.loc[df['SLA_Breached'].notna()].copy()
def breach_by_office(df):
    d=eligible_labels(df)
    if d.empty: return pd.DataFrame(columns=['Assigned_Office','Eligible_Requests','Breaches','Breach_Rate_Percent'])
    g=d.groupby('Assigned_Office',dropna=False).agg(Eligible_Requests=('SLA_Breached','size'),Breaches=('SLA_Breached','sum'),Breach_Rate=('SLA_Breached','mean')).reset_index(); g['Breach_Rate_Percent']=100*g['Breach_Rate']; return g.sort_values('Breach_Rate_Percent',ascending=False)
def breach_by_queue(df):
    d=eligible_labels(df)
    if d.empty: return pd.DataFrame(columns=['Queue_Category','Eligible_Requests','Breaches','Breach_Rate_Percent'])
    g=d.groupby('Queue_Category',observed=False,dropna=False).agg(Eligible_Requests=('SLA_Breached','size'),Breaches=('SLA_Breached','sum'),Breach_Rate=('SLA_Breached','mean')).reset_index(); g['Breach_Rate_Percent']=100*g['Breach_Rate']; return g
def volume_chart(df): return px.bar(service_volume(df),x='Service_Type',y='Request_Count',title='Requests by Service Type',labels={'Service_Type':'Service Type','Request_Count':'Requests'})
def office_chart(df): return px.bar(breach_by_office(df),x='Assigned_Office',y='Breach_Rate_Percent',title='SLA Breach Rate by Office',labels={'Assigned_Office':'Assigned Office','Breach_Rate_Percent':'Breach rate (%)'})
def queue_chart(df): return px.bar(breach_by_queue(df),x='Queue_Category',y='Breach_Rate_Percent',title='SLA Breach Rate by Queue Category',labels={'Queue_Category':'Queue Category','Breach_Rate_Percent':'Breach rate (%)'})
