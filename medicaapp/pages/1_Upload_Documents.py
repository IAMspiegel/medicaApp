import os
from typing import List
from io import StringIO
import streamlit as st

from utils import read_config, store_doc
from nlp_pipe import token_classification_pipe, llm_token_clsf_pipe, parse_date, dummy_pipe


if 'cfg' not in st.session_state:
    st.session_state['cfg'] = read_config(os.environ['APP_CONFIG'])



def process_documents(pat: str, model_type: str, files: List, container):
    for uploaded_file in files:

        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        string_data = stringio.read()

        txt_lines = []
        for line in string_data.replace('\u2028', '\n').split('\n'):
            txt_lines.append(line)

        # extract date from file
        file_date = parse_date(txt_lines)

        # NLP process
        if model_type == st.session_state['cfg']['token_classifier']['model_label']:
            medica_doc = token_classification_pipe(
                txt_lines,
                st.session_state['cfg']['token_classifier']['model_id'],
                pat,
                st.session_state['cfg']['token_classifier']['entity_mapping']
            )
        elif model_type == st.session_state['cfg']['llm']['model_label']:
            medica_doc = llm_token_clsf_pipe(
                txt_lines,
                st.session_state['cfg']['llm']['model_id'],
                pat,
                st.session_state['cfg']['llm']['entity_mapping']
            )
        else:
            medica_doc = dummy_pipe(txt_lines, pat)

        # store file
        store_doc(subject=pat, doc_date=file_date, medica_doc=medica_doc)
    
    container.success(f"Successfully processed {len(files)} files.")
    


st.set_page_config(page_title="Import Documents")

st.header("Import Documents", divider="red")

alert_container = st.container()

patients = os.listdir(os.environ["app_data"])
pat_select = st.selectbox("Select an existing subject or create a new one", patients + ['New Subject ...'])

if pat_select == 'New Subject ...':
    pat_id = st.text_input('Subject:', '')
else:
    pat_id = pat_select

model_type = st.radio(
    "NLP Pipeline",
    [
        st.session_state['cfg']['token_classifier']['model_label'],
        st.session_state['cfg']['llm']['model_label'],
        "no NLP model"
    ],
    captions=["fine tuned with BRONCO annotations", "fine tuned with different datasets", "upload documents without entity recognition"]
)

uploaded_files = st.file_uploader("Select medical reports", accept_multiple_files=True)

if st.button("Process files", type="primary", disabled=False if uploaded_files else True):
    process_documents(pat_id, model_type, uploaded_files, alert_container)
