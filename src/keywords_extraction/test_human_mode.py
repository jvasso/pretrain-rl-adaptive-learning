import sys
from .classes.keywords_extractor import KeywordsExtractor

if __name__ == "__main__":
    if len(sys.argv) > 1:
        openai_api_key = sys.argv[1]
    else:
        openai_api_key = None

systems_path = "./src/keywords_extraction/prompts/templates/system_templates/"
humans_path = "./src/keywords_extraction/prompts/templates/human_templates/"

keywords_extractor = KeywordsExtractor(system_template=f"{systems_path}template1.txt",
                                       human_template=f"{humans_path}template1.txt",
                                       llm_type="human")
keywords_extractor_list = [keywords_extractor]

KeywordsExtractor.compare_performance(keywords_extractor_list, comparison_metric="f1_score")