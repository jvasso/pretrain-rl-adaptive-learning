import sys
from .classes.keywords_extractor import KeywordsExtractor

if __name__ == "__main__":
    if len(sys.argv) > 1:
        openai_api_key = sys.argv[1]
    else:
        openai_api_key = None


templates_path = "./src/keywords_extraction/prompts/templates/llm_templates/"
systems_path   = "./src/keywords_extraction/prompts/templates/system_templates/"
humans_path    = "./src/keywords_extraction/prompts/templates/human_templates/"

gpt4all_path = "./llm_models/orca-mini-3b-gguf2-q4_0.gguf"
gpt4all_model = "orca-mini-3b-gguf2-q4_0.gguf"

configs = [
    {"llm_type":"gpt4all", "model_name":gpt4all_model, "model_path":gpt4all_path, "template_file":f"{templates_path}template1.txt"},
    {"llm_type":"gpt4all", "model_name":gpt4all_model, "model_path":gpt4all_path,  "template_file":f"{templates_path}template2.txt" },

    {"llm_type":"openai", "model_name":"gpt4", "system_template":f"{systems_path}template1.txt", "human_template":f"{humans_path}template1.txt"},
    {"llm_type":"openai", "model_name":"gpt4", "system_template":f"{systems_path}template2.txt", "human_template":f"{humans_path}template1.txt"},

    {"llm_type":"openai", "model_name":"gpt-3.5", "system_template":f"{systems_path}template1.txt", "human_template":f"{humans_path}template1.txt"},
    {"llm_type":"openai", "model_name":"gpt-3.5", "system_template":f"{systems_path}template2.txt", "human_template":f"{humans_path}template1.txt"}
]

keywords_extractors_list = []
for config in configs:
    keywords_extractor = KeywordsExtractor(openai_api_key=openai_api_key, **config)
    keywords_extractors_list.append(keywords_extractor)

KeywordsExtractor.compare_performance(keywords_extractors_list, comparison_metric="f1_score")