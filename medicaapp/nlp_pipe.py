import os
import json
import re
from dateutil.parser import parse, ParserError

import torch
from transformers import pipeline, AutoTokenizer
from peft import AutoPeftModelForCausalLM

from utils import get_file_number
from document import MedicaDoc, Section
from typing import List, Optional
    
    
def dummy_pipe(text_data: List[str], subject: str) -> MedicaDoc:
    doc_date = parse_date(text_data)
    # read into sections
    medica_doc = MedicaDoc(date=doc_date, patient=subject)
    medica_doc.feed_textlines(text_data)

    return medica_doc


def token_classification_pipe(text_data: List[str], model: str, subject: str, entity_mapping: Optional[dict]) -> MedicaDoc:
    doc_date = parse_date(text_data)
    # read into sections
    medica_doc = MedicaDoc(date=doc_date, patient=subject)
    medica_doc.feed_textlines(text_data)

    # run ner pipeline
    ner_pipe = pipeline('ner', model, aggregation_strategy='max')
    # get entities for sections
    for section in medica_doc.sections:
        # predict entities
        entities = ner_pipe(section.get_content())
        if entities:
            if entity_mapping:
                for ent in entities:
                    ent_group = entity_mapping.get(ent['entity_group'])
                    ent['entity_group'] = ent_group if ent_group else ent['entity_group']
            # append to section
            section.add_entities(entities)
    
    return medica_doc
    

def llm_token_clsf_pipe(text_data: List[str], model_id: str, subject: str, entity_mapping: Optional[dict]) -> MedicaDoc:
    doc_date = parse_date(text_data)
    # read into sections
    medica_doc = MedicaDoc(date=doc_date, patient=subject)
    medica_doc.feed_textlines(text_data)


    # load model
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_id,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        load_in_4bit=True,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # get entities for sections
    for section in medica_doc.sections:
        section_txt = section.get_content()
        # build prompt
        prompt = alpaca_instruction_prompt({"input": section_txt})
        # generate response
        output_str =_generate_llm_response(model, tokenizer, prompt)
        # prase response and build entities
        parsed_entities = parse_output(output_str, ['Medikamente', 'Behandlung', 'Diagnose'])
        cmplt_entitites = []
        for k, v in parsed_entities.items():
            if v:
                for ent in v:
                    cmplt_entitites += get_entity_offsets(ent, k, section_txt)

        # append to sectio object
        if entity_mapping:
            for ent in cmplt_entitites:
                    ent_group = entity_mapping.get(ent['entity_group'])
                    ent['entity_group'] = ent_group if ent_group else ent['entity_group']
        section.add_entities(cmplt_entitites)
    
    return medica_doc


def _generate_llm_response(model, tokenizer, prompt) -> str:
    # generate response
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=248,
        do_sample=True,
        top_p=0.95,
        temperature=0.05
    )
    output_str = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]
    return output_str


def alpaca_instruction_prompt(sample: dict) -> str:
    return f"""### Instruction:
Extrahiere die Medikamente, Behandlung und Diagnosen aus dem folgendem Text. Gib die gefundenen Entitäten als Liste aus.

### Input:
{sample['input']}

### Response:
"""


def parse_date(text_data: List[str]) -> str:
    # try to find `Entlassdatum`
    entlass_date = None
    for txt_line in text_data:
        if re.search(r'\bEntlass(?:ungs)?datum\b', txt_line):
            for lstr in txt_line.split():
                try:
                    entlass_date = parse(lstr, dayfirst=True)
                except ParserError:
                    pass
    
    # first_row = text_data[0]
    # date_matches = re.finditer(r'\[\*\*.*?\*\*\]', first_row)
    # if date_matches:
    #     for m in date_matches:
    #         return m.group()[3:-3]

    return entlass_date.strftime("%Y-%m-%d") if entlass_date else 'date(NA)'


def get_entity_offsets(ent: str, ent_group: str, section_txt: str) -> List[dict]:
    entity_dicts = []
    
    for ent_match in re.finditer(ent, section_txt):
        entity_dicts.append({
            'entity_group': ent_group,
            'word': ent_match.group(),
            'start': ent_match.span()[0],
            'end': ent_match.span()[1]
        })
    
    # TODO: fuzzy matching

    return entity_dicts


def _read_file(path) -> List[str]:
    # read file
    with open(path, 'r') as fh:
        txt_lines = []
        for line in fh.readlines():
            txt_lines.append(line)
    return txt_lines


def parse_output(output: str, entities: List[str], early_break: bool = False) -> dict:
    """Method to parse instruction finetuned output"""
    entities_dict: dict = {}

    for line in output.split('\n'):
        for ent in entities:
            if line.strip().startswith(ent):
                # quick workaround
                if ent == 'Diagnose' and line.startswith('Diagnosen'):
                    ent_output = line.replace('Diagnosen:', '').replace('\n', '')
                else:
                    ent_output = line.replace(f'{ent}:', '').replace('\n', '')
                if ent_output.strip():
                    entities_dict[ent] = [e.strip() for e in ent_output.split(',')]
                else:
                    entities_dict[ent] = None
        # check if all entities were found
        if early_break:
            if all([e in entities_dict.keys() for e in entities]):
                break
    return entities_dict
