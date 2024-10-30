import io
import os
from typing import List, Set, Union, Dict
from ast import literal_eval
from langchain.schema import BaseOutputParser
from datetime import datetime

from ...path_manager import PathManager

# wikipedia_entities_file_path = os.path.join(PathManager.WIKIPEDIA_ENTITIES_PATH, "entities.txt")
# wikipedia_entities = dict()

# with io.open(wikipedia_entities_file_path, "r", encoding="utf-8") as fp:
#     for line in fp:
#         line = line.strip()
#         wikipedia_entities[line.lower()] = line

VERBOSE = 4

class CustomOutputParser(BaseOutputParser[List[str]]):
    """Parse the output of an LLM call to a list."""

    def parse(self, text: str) -> List[str]:
        """Parse the output of an LLM call."""
        if VERBOSE >= 2: print(f'Raw output:\n{text}')
        last_list_str = self.detect_llm_final_answer(text)
        output_keywords_list = CustomOutputParser.string_to_list(last_list_str)
        final_keywords_list = self.post_process_keywords_list(output_keywords_list)
        
        if VERBOSE >= 1: print(f'\nFiltered keywords:\n{final_keywords_list}')
        return final_keywords_list
    

    def post_process_keywords_list(self, raw_keywords_list: List[str]) -> List[str]:
        keywords_list = list(set(raw_keywords_list)) # remove duplicates
        keywords_list = list(map(lambda x: x.replace(" ", "_"), keywords_list))
        
        wikipedia_entities = self.load_wikipedia_entities(mode="set")
        res_keywords = []
        if isinstance(wikipedia_entities, dict):
            for keyword in keywords_list:
                if keyword.lower() in wikipedia_entities:
                    res_keywords.append(wikipedia_entities[keyword.lower()])
        elif isinstance(wikipedia_entities, set):
            for keyword in keywords_list:
                if keyword in wikipedia_entities:
                    res_keywords.append(keyword)
        else:
            raise ValueError(f'wikipedia_entities type "{type(wikipedia_entities)}" not supported.')
        return res_keywords

    
    def detect_llm_final_answer(self, output_text:str):
        last_list_str = output_text.split("\n")[-1]
        return last_list_str
    
    
    @staticmethod
    def string_to_list(string, save=False):
        # Assuming the string is well-formed, i.e., it represents a list of strings in Python syntax
        try:
            # Convert the string representation of a list into an actual list
            # Sometimes the input is already in python list notation

            if string[0] == "[":
                actual_list = string[1:-1].split(",")
                actual_list = list(map(lambda x: x.strip(" \t\n\r'\""), actual_list))
            else:
                actual_list = string.split(',')
            if save:
                with open(f'./src/keywords_extraction/classes/outs/keywords-{datetime.now().timestamp()}.txt','w+') as f:
                    f.write(string)
            # Check if the result is indeed a list
            if not isinstance(actual_list, list):
                raise ValueError("The input does not represent a list.")
            # Check if all elements in the list are strings
            if not all(isinstance(item, str) for item in actual_list):
                raise ValueError("Not all elements are strings.")
            return actual_list
        except (SyntaxError, NameError, TypeError):
            # If there is a syntax error or a name error (which indicates the string is not a list),
            # or a type error during conversion, raise a value error.
            raise ValueError("The input is not a valid list representation.")
        
    @staticmethod
    def load_wikipedia_entities(mode="set") -> Union[Set,Dict]:
        wikipedia_entities_file_path = os.path.join(PathManager.WIKIPEDIA_ENTITIES_PATH, "entities.txt")
        if mode=="set":
            wikipedia_entities = set()
        elif mode=="dict":
            wikipedia_entities = dict()
        else:
            raise ValueError(f'Mode "{mode}" not supported.')
        
        with io.open(wikipedia_entities_file_path, "r", encoding="utf-8") as fp:
            for line in fp:
                if mode=="dict":
                    line = line.strip()
                    wikipedia_entities[line.lower()] = line
                elif mode=="set":
                    line = line.strip()
                    wikipedia_entities.add(line)
        return wikipedia_entities


if __name__=="__main__":
    import time
    t1 = time.time()
    CustomOutputParser.load_wikipedia_entities()
    print(f'Took {time.time()-t1}')



