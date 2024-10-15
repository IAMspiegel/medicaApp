import os
import io
import re
import json
from typing import List, Tuple, Optional
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as st_components

from negex.negex import *
from document import MedicaDoc, Section
from utils import aggregate_entity_info, exclude_types, timeline_plot, highlight_words, create_premed_report, premed_dict_to_markdown
from annotated_text import annotated_text
from streamlit_tags import st_tags, st_tags_sidebar


ENTITY_MAPPING = {
    'Medication': "MED",
    'Diagnosis': 'DIAG',
    'Treatment': 'TREAT'
}

PREMED_LABEL = [
    "HD",
    "ND",
    "Dauermedikation",
    "Allergien",
    "Behandlung"
]


###
# SESSION states
###
if 'selected_entities' not in st.session_state:
    st.session_state['selected_entities'] = []

if 'removed_words' not in st.session_state:
    st.session_state['removed_words'] = []

if 'df' not in st.session_state:
    st.session_state['df'] = pd.DataFrame()

def remove_entities(*words: List[str]):
    # add to session state
    st.session_state['removed_words'] += list(words)

if 'documents' not in st.session_state:
    st.session_state['documents'] = []

if 'all_entities' not in st.session_state:
    st.session_state['all_entities'] = []


# ### ### ### ### ### ### ###
#                           #
#       Methods             #
#                           #
# ### ### ### ### ### ### ###


def store_new_entity(phrase: str, label: str, cave: str, subject: str, container):
    # add new entity to dataframe
    # mapping og categories to entitiy labels
    label_mapping = {
        'HD': 'DIAG',
        'ND': 'DIAG',
        'Allergien': 'DIAG',
        'Dauermedikation': 'MED',
        'Behandlung': 'TREAT'
    }

    
    # iterate over docs
    found_entities: List[dict] = []

    #for file in os.listdir(os.path.join(os.environ["app_data"], subject)):
    for doc_idx, doc in enumerate(st.session_state['documents']):
        found_phrase_in_doc = False
        # doc = MedicaDoc.from_json(os.path.join(os.environ["app_data"], subject, file))
        # iterate over sections within document
        for sction_idx, section in enumerate(doc.sections):
            section_entities: List[dict] = []
            for phrase_match in re.finditer(phrase, section.get_content()):
                section_entities.append({
                    "entity_group": label_mapping[label],
                    "score": "1",
                    "word": phrase,
                    "start": phrase_match.start(),
                    "end": phrase_match.end()
                })

                found_entities.append({
                    "entity_group": label_mapping[label],
                    "score": "1",
                    "word": phrase,
                    "start": phrase_match.start(),
                    "end": phrase_match.end(),
                    'doc': doc.date,
                    'section_index': sction_idx,
                    'section_name': section.name,
                    'date': doc.date,
                    'negated': False,
                    'sentence': doc.extract_sentence(section.get_content(),  phrase_match.start(), phrase_match.end()),
                })
            # append to document
            if section_entities:
                section.add_entities(section_entities)
                found_phrase_in_doc = True


        # if change in document, store Doc
        if found_phrase_in_doc:
            #with open(os.path.join(os.environ["app_data"], subject, file), "w") as fh:
            #    json.dump(doc.to_dict(), fh)
            st.session_state['documents'][doc_idx] = doc
    
    if not found_entities:
        container.error(f"WARNING: cannot find new entity {phrase} in documents!")

    else:
        container.success("Added Entity")
        # get aggregated info
        agg_entity = aggregate_entity_info(found_entities).get(phrase)
        # add other info
        agg_entity['premed_category'] = label
        agg_entity['count'] = len(agg_entity['docs'])
        agg_entity['cave'] = bool(cave)
        agg_entity['select'] = False
        agg_entity['delete'] = False


        # append to dataframe
        st.session_state['df'] = pd.concat(
            [st.session_state['df'], pd.DataFrame([agg_entity])],
            ignore_index=True
        )
        #st.cache_data.clear()



def get_premed_dict() -> dict:
    ## collect entities for report and create dict
    df_selected = st.session_state['df'][ st.session_state['df']['select'] == True].copy()
    premed_dict = df_selected.groupby('premed_category')['word'].apply(list).to_dict()
    
    # select cave entitites
    premed_dict['cave'] = df_selected[df_selected['cave'] == True]['word'].to_list()

    return premed_dict

def create_report():
    premed_dict = get_premed_dict()
    # # insert entities into template
    report_document = create_premed_report(premed_dict, st.session_state['pat_select'])
    report_document.save('stage/testPreMedReport.docx')


def update_dataframe(df_show: pd.DataFrame):
    """Method to update base dataframe in sessions state"""
    for idx, edited_row in st.session_state['edited_df']['edited_rows'].items():
        for label, value in edited_row.items():
            st.session_state['df'].loc[df_show.iloc[idx].name, label] = value

#@st.cache_data
def collect_subject_data(subject: str) -> Tuple[List[MedicaDoc], List[dict], dict, pd.DataFrame, List[dict]]:
    """Method to read files for selected subject and build objects, aggregate entities and dataframe"""
    # init german negex rules
    rfile = open(r'../negex/negex_trigger_german_biotxtm_2016.txt')
    irules = sortRules(rfile.readlines())
    # collect document objects from files
    documents: List[MedicaDoc] = []
    for file in os.listdir(os.path.join(os.environ["app_data"], subject)):
        documents.append(
            MedicaDoc.from_json(os.path.join(os.environ["app_data"], subject, file))
        )
    # build entities
    all_entities: List[dict] = []
    for doc in documents:
        all_entities += doc.get_all_entities()
    # negation detection
    for ent in all_entities:
        tagger = negTagger(ent['sentence'], [ent['word']], rules=irules)
        if tagger.getNegationFlag() == 'negated':
            ent['negated'] = True
        else:
           ent['negated'] = False
    # aggregate entities
    agg_entities = aggregate_entity_info(all_entities)
    # build dataframe
    if agg_entities:
        df_agg = pd.DataFrame.from_records([v for _, v in agg_entities.items()])
        df_agg['count'] = df_agg['docs'].apply(lambda x: len(x))
        df_agg['select'] = False
        df_agg['delete'] = False
        df_agg['cave'] = False
        df_agg['premed_category'] = None
        df_agg.loc[df_agg['negated'] == True, 'negation'] = 'Neg'
    else:
        df_agg = pd.DataFrame(
            columns=['docs', 'count', 'select', 'delete', 'cave', 'premed_category',
                     'negated', 'type', 'word', 'negation', 'first_occurance', 'last_occurance']
        )

    return documents, df_agg, all_entities


def change_patient(initial_pat: Optional[str] = None):
    if initial_pat:
        patient = initial_pat
    else:
        patient = st.session_state['pat_select']

    # collect patient data
    documents, df_agg, all_entities = collect_subject_data(patient)

    # add to session state
    st.session_state['df'] = df_agg
    # set session state with entities
    st.session_state['all_entities'] = all_entities
    # set docs
    st.session_state['documents'] = documents


# ### ### ### ### ### ### ###
#                           #
#   Start of page build     #
#                           #
# ### ### ### ### ### ### ###

patients = os.listdir(os.environ["app_data"])
st.set_page_config(page_title="Subject", layout="wide")


# for initial page lode -> change_patient() method was not triggered
if not st.session_state['documents']:
    initial_pat = patients[0]
    change_patient(initial_pat=initial_pat)


doc_timeline: List[dict] = []
doc_tabs = []
for i, doc in enumerate(st.session_state['documents'] ):
    # append to list
    doc_tabs.append(f"Doc_{doc.date}")
    doc_timeline.append(datetime.strptime(doc.date, '%Y-%m-%d'))



# -----------------------------
# HEADER

st.header("Premedication", divider="red")


st.subheader(f"Subject: {st.session_state['pat_select'] if 'pat_select' in st.session_state else patients[0]}")

checks = st.columns([1, 1, 1, 3])
with checks[0]:
    check_med = st.checkbox('Medication', value=True)
with checks[1]:
    check_diag = st.checkbox('Diagnosis', value=True)
with checks[2]:
    check_treat = st.checkbox('Treatment', value=True)


checks_two = st.columns([1.5, 1.5, 4])
with checks_two[0]:
    check_neg = st.checkbox('Exclude Negations', value=False)
with checks_two[1]:
    check_selected_only = st.checkbox(
        'Show only selected',
        value=False,
        disabled=False if st.session_state['selected_entities'] else True
    )

# -----------------------------
# Tabs

tab_containers = st.tabs(["Overview"] + doc_tabs + ['PreMed Report'])

tab_overview = tab_containers[0]

excluded_types = []


# -----------------------------
# Overview Table

with tab_overview:
    st.subheader("Found entities")

    #if ENTITY_MAPPING
    if not check_med:
        excluded_types.append('MED')
    else:
        if 'MED' in excluded_types:
            excluded_types.remove('MED')
    if not check_diag:
        excluded_types.append('DIAG')
    else:
        if 'DIAG' in excluded_types:
            excluded_types.remove('DIAG')
    if not check_treat:
        excluded_types.append('TREAT')
    else:
        if 'TREAT' in excluded_types:
            excluded_types.remove('TREAT')

    df_show = exclude_types(st.session_state['df'].copy(), excluded_types).copy()
    if check_neg:
        df_show = df_show[df_show['negation'] != 'Neg']
    if check_selected_only:
        df_show = df_show[df_show['select'] == True]

    # remove words
    if st.session_state['removed_words']:
        df_show = df_show[~df_show['word'].isin(st.session_state['removed_words'])]
    # select words in detail view
    if st.session_state['selected_entities']:
        df_show.loc[df_show['word'].isin(st.session_state['selected_entities']), 'select'] = True

    st.data_editor(
            df_show[['type', 'select', 'word', 'premed_category', 'cave','negation','first_occurance', 'last_occurance', 'count', 'delete']],
            column_config={
                'select': st.column_config.CheckboxColumn(required=True, width="small"),
                'premed_category': st.column_config.SelectboxColumn(options=PREMED_LABEL + [None], default=None, width="medium"),
                'delete': st.column_config.CheckboxColumn(width="small"),
                'cave': st.column_config.CheckboxColumn(width="small")
            },
            disabled=[c for c in st.session_state['df'].columns if c not in ['type', 'select', 'delete', 'cave', 'premed_category']],
            use_container_width=True,
            key='edited_df',
            on_change=update_dataframe,
            kwargs={'df_show': df_show}
        )

    selected_words = st.session_state['df'][st.session_state['df']['select'] == True]['word'].to_list()

    # update session state
    displayed_entities = list(df_show['word'].unique())
    # check if entity was de-selected
    for ent in st.session_state['selected_entities']:
        if ent not in selected_words and ent in displayed_entities:
            st.session_state['selected_entities'].remove(ent)
    # update new added word
    for ent in selected_words:
        if ent not in st.session_state['selected_entities']:
            st.session_state['selected_entities'].append(ent)

    # entities selceted to delete
    deleted_words = st.session_state['df'][st.session_state['df']['delete'] == True]['word'].to_list()

    if st.session_state['selected_entities']:
        # collect documents for selected entities
        docs_with_selected_ents: dict = {}
        for ent in st.session_state["all_entities"]:
            if ent['word'] in st.session_state['selected_entities']:
                # check for doc
                if ent['doc'] in docs_with_selected_ents:
                    # check for section
                    if ent['sentence'] in docs_with_selected_ents[ent['doc']]:
                        docs_with_selected_ents[ent['doc']][ent['sentence']].append(ent)
                    else:
                        docs_with_selected_ents[ent['doc']][ent['sentence']] = [ent]
                else:
                    docs_with_selected_ents[ent['doc']] = {ent['sentence']: [ent]}
        # fill container
        ent_container = st.container()
        ent_container.subheader("Detailed view")
        for doc, sections in docs_with_selected_ents.items():
            with ent_container.expander(f'Doc ({doc}):'):
                for _, sect_ents in sections.items():
                    st.write(highlight_words(sect_ents))

    if deleted_words:
        selected_delete_words = st_tags_sidebar(deleted_words, label="Delete selected entites:")
        st.sidebar.button("Delete", type="primary", on_click=remove_entities, args=selected_delete_words)


# -----------------------------
# Document Tabs

# build other tabs
for mdoc, t_container in zip(st.session_state['documents'], tab_containers[1:-1]):
    with t_container:
        st.subheader("Dokument:")
        for section in mdoc.sections:
            annotated_text(
                *section.to_annotations_list(
                    exclude_types=excluded_types,
                    entities=st.session_state['selected_entities'] if check_selected_only else None,
                    filter_negations=check_neg
                )
            )


# -----------------------------
# PreMed Report tab

# overview tab of report and download button
with tab_containers[-1]:
    if os.path.isfile('stage/testPreMedReport.docx'):
        with open(r'stage/testPreMedReport.docx', 'rb') as file:
            st.download_button("Download PreMed Report", type="primary", data=file, file_name="TestPreMedReport.docx")
        
        #st.write(pre_report)
        st.markdown(premed_dict_to_markdown(get_premed_dict()))
    else:
        st.write('No created Report.')


# -----------------------------
# Sidebar

with st.sidebar:
    pat = st.selectbox(
        "Select patient:",
        patients,
        key="pat_select",
        on_change=change_patient
    )

    if st.session_state['selected_entities']:
        # collect documents for selected entities
        selected_sidebar_words = st_tags_sidebar(st.session_state['selected_entities'], label="Selected entites:")
        st.session_state['selected_entities'] = selected_sidebar_words

        st.sidebar.button("Create Report", on_click=create_report, type='primary')

    with st.expander("Create Entity"):
        new_entity_phrase = st.text_input("New Entity")
        new_entity_label = st.selectbox("PreMed Entity", PREMED_LABEL, index=None)
        new_entity_cave = st.checkbox("Cave")
        container = st.container()
        container.button("Store Entity", on_click=store_new_entity, kwargs={"phrase": new_entity_phrase, "label": new_entity_label, "cave": new_entity_cave, "subject": pat, "container": container}, type='primary')
