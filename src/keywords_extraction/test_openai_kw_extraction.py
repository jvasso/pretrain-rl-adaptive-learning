import sys
from .classes.keywords_extractor import KeywordsExtractor

if __name__ == "__main__":
    if len(sys.argv) > 1:
        openai_api_key = sys.argv[1]
    else:
        openai_api_key = None

systems_path   = "./src/keywords_extraction/prompts/templates/system_templates/"
humans_path    = "./src/keywords_extraction/prompts/templates/human_templates/"

configs = [
    # {"llm_type":"openai", "model_name":"gpt-3.5-turbo", "system_template":f"{systems_path}template1.txt", "human_template":f"{humans_path}template1.txt"},
    # {"llm_type":"openai", "model_name":"gpt-3.5-turbo", "system_template":f"{systems_path}template2.txt", "human_template":f"{humans_path}template1.txt"},
    # {"llm_type":"openai", "model_name":"gpt-3.5-turbo", "system_template":f"{systems_path}template3.txt", "human_template":f"{humans_path}template1.txt"},
    # dict(llm_type="openai", model_name="gpt_3_5_turbo_api", system_template=f"{systems_path}template3.txt", human_template=f"{humans_path}template1.txt", use_azure=True),
    # dict(llm_type="openai", model_name="gpt4", system_template=f"{systems_path}template2.txt", human_template=f"{humans_path}template1.txt", use_azure=True, temperature=0, max_tokens=1000),
    # dict(llm_type="openai", model_name="gpt4", system_template=f"{systems_path}template3.txt", =f"{humans_path}template1.txt", use_azure=True, temperature=0, max_tokens=1000),
    # dict(llm_type="openai", model_name="gpt4", system_template=f"{systems_path}template4.txt", human_template=f"{humans_path}template1.txt", use_azure=True, temperature=0, max_tokens=1000),
    # dict(llm_type="openai", model_name="gpt4", system_template=f"{systems_path}template5.txt", human_template=f"{humans_path}template1.txt", use_azure=True, temperature=0, max_tokens=1000),
    dict(llm_type="openai", model_name="gpt4", system_template=f"{systems_path}template6.txt", human_template=f"{humans_path}template1.txt", use_azure=True, temperature=0, max_tokens=1000),
    dict(llm_type="openai", model_name="gpt4", system_template=f"{systems_path}template7.txt", human_template=f"{humans_path}template1.txt", use_azure=True, temperature=0, max_tokens=1000)
]

keywords_extractors_list = []
for config in configs:
    keywords_extractor = KeywordsExtractor(openai_api_key=openai_api_key, **config)
    keywords_extractors_list.append(keywords_extractor)

KeywordsExtractor.compare_performance(keywords_extractors_list, comparison_metric="f1_score", save_perfs=True)