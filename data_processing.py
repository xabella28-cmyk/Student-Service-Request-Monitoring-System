"""Validation and cleaning utilities aligned with the uploaded Section B notebook."""
import io
import pandas as pd
import numpy as np
REQUIRED_COLUMNS = ["Request_ID","Service_Type","Requester_Type","Submission_Date","Requested_Channel","Priority","Assigned_Office","Queue_Length","Required_Documents_Complete","Processing_Days","SLA_Days","Status","Reopened","Satisfaction_Score"]
CATEGORICAL_COLUMNS = ["Service_Type","Requester_Type","Requested_Channel","Priority","Assigned_Office","Required_Documents_Complete","Status","Reopened"]
NUMERIC_COLUMNS = ["Queue_Length","Processing_Days","SLA_Days","Satisfaction_Score"]
def validate_dataframe(df):
    if not isinstance(df,pd.DataFrame): return False,["Input is not a pandas DataFrame."]
    missing=[c for c in REQUIRED_COLUMNS if c not in df.columns]
    return len(missing)==0, ([f"Missing required columns: {', '.join(missing)}"] if missing else [])
def read_uploaded_file(uploaded):
    name=getattr(uploaded,"name","").lower()
    try:
        if name.endswith('.csv'): df=pd.read_csv(uploaded)
        elif name.endswith(('.xlsx','.xls')): df=pd.read_excel(uploaded)
        else: return None,["Unsupported file type. Upload CSV or Excel (.xlsx/.xls)."]
    except Exception as exc: return None,[f"Could not read file: {exc}"]
    ok,errors=validate_dataframe(df)
    return (df,[]) if ok else (None,errors)
def clean_and_engineer(df):
    ok,errors=validate_dataframe(df)
    if not ok: raise ValueError('; '.join(errors))
    clean=df.copy(); log=[]
    def add(issue,count,treatment): log.append({"Issue":issue,"Records_Affected":int(count),"Treatment":treatment})
    dups=int(clean.Request_ID.duplicated(keep='first').sum()); add('Duplicate Request_ID',dups,'Kept first occurrence and removed later duplicates.'); clean=clean.drop_duplicates(subset=['Request_ID'],keep='first').copy()
    for c in CATEGORICAL_COLUMNS:
        before=clean[c].astype('string'); after=before.str.strip().str.replace(r'\s+',' ',regex=True); add(f'Inconsistent formatting: {c}',int((before.fillna('<NA>')!=after.fillna('<NA>')).sum()),'Trimmed edges and collapsed repeated whitespace.'); clean[c]=after
    for c in ['Required_Documents_Complete','Status','Reopened']: clean[c]=clean[c].str.title()
    original=clean.Submission_Date.copy(); dates=pd.to_datetime(original,errors='coerce'); add('Invalid Submission_Date',int((original.notna()&dates.isna()).sum()),'Converted invalid dates to missing values.'); clean['Submission_Date']=dates
    for c in NUMERIC_COLUMNS:
        orig=clean[c].notna(); clean[c]=pd.to_numeric(clean[c],errors='coerce'); add(f'Non-numeric values: {c}',int((orig&clean[c].isna()).sum()),'Converted unparseable values to missing.')
        neg=clean[c]<0; add(f'Negative values: {c}',int(neg.sum()),'Replaced negative values with missing values.'); clean.loc[neg,c]=np.nan
    clean['Submission_Weekday']=clean.Submission_Date.dt.day_name()
    clean['Queue_Category']=pd.cut(clean.Queue_Length,bins=[-np.inf,5,15,np.inf],labels=['Low','Medium','High'])
    eligible=clean.Status.eq('Completed')&clean.Processing_Days.notna()&clean.SLA_Days.notna()
    clean['SLA_Breached']=pd.Series(pd.NA,index=clean.index,dtype='Int64')
    clean.loc[eligible,'SLA_Breached']=(clean.loc[eligible,'Processing_Days']>clean.loc[eligible,'SLA_Days']).astype(int)
    for c in clean.columns:
        n=int(clean[c].isna().sum())
        if n: add(f'Missing values: {c}',n,'Retained missingness; predictors are imputed inside the training pipeline and unlabeled rows are excluded from model training.')
    return clean,pd.DataFrame(log)
