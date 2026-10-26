import streamlit as st
import pandas as pd
from pathlib import Path
from modules.data_processing import read_uploaded_file,clean_and_engineer,REQUIRED_COLUMNS
from modules.exploratory_analysis import service_volume,breach_by_office,breach_by_queue,volume_chart,office_chart,queue_chart
from modules.prediction import load_model,predict_one
from modules.dashboard import dashboard_summary,flagged_queue
ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='Student Service Request Monitoring',page_icon='📊',layout='wide')
@st.cache_data
def load_default_data():
    p=ROOT/'data'/'cleaned_student_service_requests.csv'
    return pd.read_csv(p) if p.exists() else pd.DataFrame()
@st.cache_resource
def get_bundle(): return load_model()
def banner(): st.caption('Decision-support prototype • Predictions are estimates, not confirmed outcomes. Authorized human review is required.')
def page_home(df):
    st.title('Student Service Request Monitoring System'); banner()
    st.write('This app summarizes historical student service requests, explores service-level agreement (SLA) patterns, and provides an advisory decision-tree prediction based only on information available at submission time.')
    s=dashboard_summary(df) if len(df) else {'total_requests':0,'eligible_completed':0,'breach_rate_percent':None,'pending_requests':0,'high_queue_requests':0}
    c=st.columns(4); c[0].metric('Total requests',s['total_requests']); c[1].metric('Eligible labeled records',s['eligible_completed']); c[2].metric('Pending requests',s['pending_requests']); c[3].metric('High-queue requests',s['high_queue_requests'])
    st.info('Model target: SLA_Breached = 1 when a completed request took longer than its SLA; 0 otherwise. Open/cancelled requests are not labeled for training.')
    st.subheader('System capabilities'); st.markdown('- Upload and validate CSV/Excel files\n- Inspect data quality and exploratory charts\n- Review the Decision Tree and majority-class baseline metrics\n- Estimate breach risk for a hypothetical request\n- View system requirements and analytical workflow')
def page_upload(default):
    st.title('Dataset Upload & Validation'); banner(); st.write('Required source fields: '+', '.join(REQUIRED_COLUMNS))
    up=st.file_uploader('Upload CSV or Excel workbook',type=['csv','xlsx','xls'])
    source=default
    if up is not None:
        incoming,errors=read_uploaded_file(up)
        if errors: st.error('\n'.join(errors)); return default
        source=incoming; st.success(f'Loaded {len(source):,} rows and {len(source.columns)} columns from upload.')
    if source.empty: st.warning('No valid dataset is loaded.'); return default
    clean,log=clean_and_engineer(source)
    a,b,c=st.columns(3); a.metric('Rows after cleaning',len(clean)); b.metric('Columns',len(clean.columns)); c.metric('Quality-log items',len(log))
    st.subheader('Preview'); st.dataframe(clean.head(20),use_container_width=True)
    st.subheader('Missing values'); st.dataframe(clean.isna().sum().rename('Missing Count').to_frame(),use_container_width=True)
    st.subheader('Cleaning log'); st.dataframe(log,use_container_width=True)
    st.download_button('Download cleaned CSV',clean.to_csv(index=False).encode('utf-8'),file_name='cleaned_student_service_requests.csv',mime='text/csv')
    st.download_button('Download data-quality log',log.to_csv(index=False).encode('utf-8'),file_name='data_quality_log.csv',mime='text/csv')
    return clean

def page_eda(df):
    st.title('Exploratory Data Analysis'); banner()
    if df.empty: st.warning('Load a dataset first.'); return
    st.subheader('Requests by service type'); st.plotly_chart(volume_chart(df),use_container_width=True); st.dataframe(service_volume(df),use_container_width=True)
    st.subheader('SLA breach rate by assigned office'); off=breach_by_office(df); st.plotly_chart(office_chart(df),use_container_width=True); st.dataframe(off,use_container_width=True)
    st.subheader('SLA breach rate by queue category'); q=breach_by_queue(df); st.plotly_chart(queue_chart(df),use_container_width=True); st.dataframe(q,use_container_width=True)
    st.caption('Breach rates use only completed records with known Processing_Days and SLA_Days. Small groups should be interpreted cautiously.')

def page_model(df):
    st.title('Predictive Modeling'); banner()
    try: bundle=get_bundle()
    except Exception as e: st.error(f'Model could not be loaded: {e}'); return
    st.write(f"Model version: {bundle.get('model_version','unknown')} | Algorithm: Decision Tree | random_state: {bundle.get('training_random_state',42)}")
    st.subheader('Holdout evaluation'); ev=ROOT/'reports'/'model_evaluation.csv'; cm=ROOT/'reports'/'confusion_matrix.csv'; imp=ROOT/'reports'/'feature_importance.csv'
    if ev.exists(): st.dataframe(pd.read_csv(ev),use_container_width=True)
    if cm.exists(): st.write('Confusion matrix (rows = actual; columns = predicted)'); st.dataframe(pd.read_csv(cm,index_col=0),use_container_width=True)
    if imp.exists():
        idf=pd.read_csv(imp); st.subheader('Feature importance'); st.dataframe(idf.head(10),use_container_width=True)
        st.bar_chart(idf.head(10).set_index('Feature')['Importance'])
    st.warning('A model score is not a guarantee. Do not automatically deny, deprioritize, or penalize a requester based on a prediction.')
    st.subheader('Predict a hypothetical request')
    with st.form('predict_form'):
        cols=st.columns(2); vals={}
        choices={'Service_Type':['Enrollment','Records','Certification','IT Support','Scholarship Inquiry'],'Requester_Type':['Student','Faculty','Staff'],'Requested_Channel':['Portal','Email','Walk-in'],'Priority':['Low','Normal','High'],'Assigned_Office':['Student Affairs','Registrar','IT Services','Scholarship Office'],'Required_Documents_Complete':['Yes','No'],'Submission_Weekday':['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],'Queue_Category':['Low','Medium','High']}
        for i,f in enumerate(bundle['features']):
            with cols[i%2]:
                if f in choices: vals[f]=st.selectbox(f.replace('_',' '),choices[f])
                elif f=='Queue_Length': vals[f]=st.number_input('Queue Length',min_value=0,value=5,step=1)
                elif f=='SLA_Days': vals[f]=st.number_input('SLA Days',min_value=0.0,value=5.0,step=1.0)
                else: vals[f]=None
        go=st.form_submit_button('Estimate SLA breach risk')
    if go:
        try:
            result=predict_one(vals,bundle); st.metric('Predicted class','Potential SLA breach' if result['prediction']==1 else 'No breach predicted')
            if result['probability'] is not None: st.metric('Estimated breach probability',f"{result['probability']:.1%}")
            st.write('Submission-time inputs used'); st.dataframe(result['input'],use_container_width=True)
            st.info('Use this result to prompt a staff review. It is not a confirmed outcome or a reason for automatic adverse action.')
        except Exception as e: st.error(f'Prediction failed: {e}')
    five=ROOT/'reports'/'five_fictional_predictions.csv'
    if five.exists():
        st.subheader('Five fictional demonstration cases'); st.dataframe(pd.read_csv(five),use_container_width=True)

def page_dashboard(df):
    st.title('Decision-Support Dashboard'); banner()
    if df.empty: st.warning('Load a dataset first.'); return
    s=dashboard_summary(df); c=st.columns(4); c[0].metric('Total requests',s['total_requests']); c[1].metric('Eligible completed',s['eligible_completed']); c[2].metric('Pending',s['pending_requests']); c[3].metric('High queue',s['high_queue_requests'])
    if s['breach_rate_percent'] is not None: st.metric('Historical SLA breach rate',f"{s['breach_rate_percent']:.1f}%")
    left,right=st.columns(2); left.plotly_chart(volume_chart(df),use_container_width=True); right.plotly_chart(office_chart(df),use_container_width=True)
    st.subheader('Requests for operational review'); st.dataframe(flagged_queue(df),use_container_width=True)
    st.caption('The review queue is a simple queue-based view, not a prediction of individual open-request outcomes.')
def page_design():
    st.title('System Design & Requirements'); banner()
    st.subheader('Actors'); st.markdown('- **Administrator:** reviews service volume, SLA performance, queue status, and model evaluation.\n- **Service Staff:** reviews request details and uses risk estimates as one input to prioritization.\n- **System:** validates files, cleans data, engineers features, produces analytics, and estimates risk.')
    st.subheader('Functional requirements'); st.markdown('1. The system shall validate uploaded data against required service-request fields.\n2. The system shall display service volume, eligible SLA breach rates by office, and queue category.\n3. The system shall estimate SLA breach risk from submission-time fields only.\n4. The system shall show model evaluation metrics and feature importance.\n5. The system shall present risk estimates for authorized human review, not automatic denial.')
    st.subheader('Non-functional requirements'); st.markdown('- **Privacy:** restrict access to authorized staff and avoid unnecessary personal data.\n- **Reliability:** show validation errors and missing-data counts.\n- **Reproducibility:** fixed random_state=42 for the train/test split and Decision Tree.\n- **Usability:** labels, tables, and charts use understandable names.\n- **Safety:** predictions are advisory and require human review.')
    st.subheader('UML Use Case Diagram'); p=ROOT/'assets'/'uml_use_case.png'
    if p.exists(): st.image(str(p),use_container_width=True)
    st.subheader('Analytical process flow'); p=ROOT/'assets'/'process_flow.png'
    if p.exists(): st.image(str(p),use_container_width=True)
    st.subheader('Data dictionary'); st.dataframe(pd.DataFrame([{'Field':'Request_ID','Meaning':'Unique transaction identifier','Use':'Identifier only; not a predictor'},{'Field':'Service_Type','Meaning':'Requested service category','Use':'Predictor'},{'Field':'Requester_Type','Meaning':'Student, Faculty, or Staff','Use':'Predictor'},{'Field':'Submission_Date','Meaning':'Date request was lodged','Use':'Derive weekday'},{'Field':'Requested_Channel','Meaning':'Portal, Email, Walk-in','Use':'Predictor'},{'Field':'Priority','Meaning':'Recorded priority at intake','Use':'Predictor'},{'Field':'Assigned_Office','Meaning':'Office processing request','Use':'Predictor'},{'Field':'Queue_Length','Meaning':'Queue size at submission','Use':'Predictor'},{'Field':'Required_Documents_Complete','Meaning':'Whether required documents are complete','Use':'Predictor'},{'Field':'Processing_Days','Meaning':'Actual elapsed processing duration','Use':'Target construction only; leakage if predictor'},{'Field':'SLA_Days','Meaning':'Allowed processing duration','Use':'Predictor'},{'Field':'Status','Meaning':'Completed, Pending, Cancelled','Use':'Eligibility filter only'},{'Field':'Reopened','Meaning':'Whether request was reopened','Use':'Excluded from predictors'},{'Field':'Satisfaction_Score','Meaning':'Post-service rating','Use':'Excluded from predictors'},{'Field':'SLA_Breached','Meaning':'1 if completed duration exceeds SLA; else 0','Use':'Target; only eligible completed records'}]),use_container_width=True)

def main():
    st.sidebar.title('Navigation'); page=st.sidebar.radio('Choose a page',['Home','Dataset Upload & Validation','Exploratory Data Analysis','Predictive Modeling','Decision-Support Dashboard','System Design'])
    st.sidebar.caption('Section B • Data Mining and Systems Analysis Integration')
    df=load_default_data()
    if not df.empty:
        # Normalize types when reading the saved CSV.
        df['Submission_Date']=pd.to_datetime(df['Submission_Date'],errors='coerce')
        df['SLA_Breached']=pd.to_numeric(df['SLA_Breached'],errors='coerce')
    if page=='Home': page_home(df)
    elif page=='Dataset Upload & Validation':
        uploaded_clean=page_upload(df)
        if not uploaded_clean.empty: st.session_state['active_data']=uploaded_clean
    elif page=='Exploratory Data Analysis': page_eda(st.session_state.get('active_data',df))
    elif page=='Predictive Modeling': page_model(st.session_state.get('active_data',df))
    elif page=='Decision-Support Dashboard': page_dashboard(st.session_state.get('active_data',df))
    else: page_design()
if __name__=='__main__': main()
