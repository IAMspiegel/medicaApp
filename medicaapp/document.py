from typing import List, Optional, Dict, Union
from dataclasses import dataclass, field
import json
import re


class MedicaDoc:
    """
    Class to represent a medical document of the `medicaApp`.

    """
    def __init__(self, date: str, patient: str, nlp_pipeline_type: Optional[str] = None, sections: Optional[List] = None):
        self.patient = patient
        self.date = date
        self.nlp_pipe_type = nlp_pipeline_type
        self.sections = sections

    def __repr__(self) -> str:
        return f"{self.patient}_{self.date}_{self.nlp_pipe_type}_{len(self.sections)}"

    @classmethod
    def from_json(cls, file_path: str, replace_dict: Optional[dict] = None):
        """Method to initiate object from json file"""
        with open(file_path, "r") as fh:
            json_data = json.load(fh)

        sections = []
        for sect in json_data['sections']:
            sections.append(Section(**sect))
        
        return cls(
            date=json_data['date'],
            patient=json_data['patient'],
            sections=sections
        )
        
    def get_entities_by_class(self, ent_class: str) -> Optional[List[dict]]:
        class_entities = []
        for sect in self.sections:
            if sect.entities.get(ent_class):
                class_entities += sect.entities.get(ent_class).copy()
        return class_entities

    def get_all_entities(self) -> Optional[List[dict]]:
        """Get all entities with infomration about entity origin"""
        # entity_dict: Dict[str, List] = {}

        # for i, sect in enumerate(self.sections):
        #     for k, v in sect.entities.items():
        #         if k in entity_dict:
        #             entity_dict[k] += v.copy()
        #         else:
        #             entity_dict[k] = v.copy()

        doc_entities: List[dict] = []

        for i, section in enumerate(self.sections):
            for ent in [ent for _, section_ents in section.entities.items() for ent in section_ents]:
                ent['doc'] = f"{self.date}"
                ent['section_index'] = i
                ent['section_name'] = section.name
                ent['date'] = self.date
                ent['sentence'] = self.extract_sentence(section.get_content(), ent['start'], ent['end'])
                doc_entities.append(ent)

        return doc_entities
    
    def get_content_with_annotations(self) -> List[Union[str, tuple]]:
        anno_content = []
        for section in self.sections:
            anno_content += section.to_annotations_list()
        return anno_content

    def feed_textlines(self, txt_lines: List[str]):
        """Method to read text file into section objects"""
        curr_section: Section = None
        curr_offset: int = 0  # keep track of text offsets from sections
        sections_before = len(self.sections) if self.sections else 0
        empty_row_count: int = 0

        for i, line in enumerate(txt_lines):
            if line:
                
                # check if line contains section start
                section_key = self._section_start(line)
                # new section after double break

                if section_key:
                    # check if current section exists
                    if not curr_section:
                        curr_section = Section(
                            name=section_key,
                            start=curr_offset,
                            nlp_pipe=self.nlp_pipe_type
                        )
                        #subsection = section_key
                        #subsection_content.append(line)

                        curr_section.add_content(line)

                    else:
                        # process section
                        if self.sections:
                            self.sections.append(curr_section)
                        else:
                            self.sections = [curr_section]
                        # create new
                        curr_section = Section(
                            name=section_key,
                            start=curr_offset,
                            nlp_pipe=self.nlp_pipe_type
                        )
                        curr_section.add_content(line)
                elif i == 0:
                    curr_section = Section(
                        name="firstLine",
                        start=curr_offset,
                        nlp_pipe=self.nlp_pipe_type
                    )
                    curr_section.add_content(line)
                # two breakes indicate new section
                elif empty_row_count >= 2:
                    # append section
                    self.sections.append(curr_section)
                    # init new section
                    curr_section = Section(
                        name="paragraph",
                        start=curr_offset,
                        nlp_pipe=self.nlp_pipe_type
                    )
                    curr_section.add_content(line)
                else:
                    curr_section.add_content(line)

                curr_offset += len(line)
                empty_row_count = 0
            else:
                empty_row_count += 1

        self.sections.append(curr_section)

        print(f'Added {len(self.sections) - sections_before} sections.')

    @staticmethod
    def _section_start(txt: str) -> Optional[str]:
        """Method to determine if txt is start of a new section"""
        substings = txt.split()
        max_title_words = 6

        if len(substings) <= max_title_words:
            max_title_words = len(substings) - 1

        if len(substings) < 5 and txt.endswith(':'):
            return txt
        
        for i in range(max_title_words):
            if substings[i].endswith(':'):
                return ' '.join(substings[:i+1])
    
    @staticmethod
    def extract_sentence(paragraph, start, end):
        # Ensure start and end are within the bounds of the paragraph
        start = max(0, min(start, len(paragraph)))
        end = min(len(paragraph), max(end, 0))
        
        # Use regular expression to find the start of the sentence
        sentence_start_match = re.search(r'[\n.!?]', paragraph[:start][::-1])
        sentence_start = start - sentence_start_match.start() if sentence_start_match else 0
        
        # Use regular expression to find the end of the sentence
        sentence_end_match = re.search(r'[\n.!?]', paragraph[end:])
        sentence_end = end + sentence_end_match.start() if sentence_end_match else len(paragraph)
        
        extracted_sentence = paragraph[sentence_start:sentence_end + 1]
        
        return extracted_sentence

    def to_dict(self) -> dict:
        return {
            "patient": self.patient,
            "date": self.date,
            "sections": [section.to_dict() for section in self.sections]
        }


@dataclass
class Section:
    name: str
    start: int
    end: int = None
    nlp_pipe: str = None
    content: List[str] = None
    entities: Dict[str, List[dict]] = field(default_factory= lambda: {})

    def add_content(self, new_content: str):

        if not isinstance(new_content, str):
            raise TypeError('Section only support string content.')
        # append to content list
        if self.content:
            self.content.append(new_content)
        else:
            self.content = [new_content]
        # update section end
        self._update_end()

    def get_content(self, use_html_break: bool = False) -> str:
        # return content list as string
        return ''.join(self.content)

    def _update_end(self):
        self.end = self.start + len(self.get_content(use_html_break=True))

    def add_entities(self, new_entities: List[dict]):
        """Add entities to section entities dictionary"""
        
        for ent in new_entities:
            ent_class = ent['entity_group']
            if ent_class in self.entities:
                self.entities[ent_class].append(ent)
            else:
                self.entities[ent_class] = [ent]

    def to_annotations_list(
            self,
            exclude_types: Optional[List[str]] = None,
            entities: Optional[List] = None,
            filter_negations: bool = False
        ) -> List[Union[str, tuple]]:
        """
        Transform section content to data format fitting to the streamlit `st-annotated-text` component
        -> ['Das Medikament ', ('Ibuprofen', 'Drug'), ' hilft.']
        """
        anno_text = []
        base_text = self.get_content(use_html_break=True)
        latest_offset = 0

        if self.entities:
            # filter entities
            filtered_entities = []
            for k, v in self.entities.items():
                # check type
                if (exclude_types and k not in exclude_types) or not exclude_types:
                    # check selected entity
                    for ent in v:
                        if (not entities or ent['word'] in entities) and (not filter_negations or not ent['negated']):
                            filtered_entities.append(ent)
            # build text
            #for ent in sorted([e for _, v in self.entities.items() for e in v], key=lambda x: x['start']):
            for ent in sorted(filtered_entities, key=lambda x: x['start']):
                # first add text before to txt
                anno_text.append(base_text[latest_offset:(ent['start'])])
                # build tuple
                anno_text.append((base_text[ent['start']:ent['end']], ent['entity_group']))
                latest_offset = ent['end']
            if latest_offset != len(base_text) - 1:
                anno_text.append(base_text[latest_offset:])
        else:
            for c in self.content:
                anno_text.append(c + '  \n')
        return anno_text
    
    def to_dict(self) -> dict:
        """Returns json conform dict"""
        json_conform_ents: Dict = {}
        if self.entities:
            for k, entities in self.entities.items():
                # transform float to string
                json_ents = []
                for ent in entities:
                    if 'score' in ent:
                        ent['score'] = str(ent['score'])
                    json_ents.append(ent)
                # append to dict
                json_conform_ents[k] = json_ents

        return {
                "name": self.name,
                "start": self.start,
                "end": self.end,
                "content": self.content,
                "entities": json_conform_ents
            }
