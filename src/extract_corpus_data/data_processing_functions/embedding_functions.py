import os
import time
import datetime
import json

import numpy as np

from ... import utils

from gensim.models import KeyedVectors
# import tagme
# tagme.GCUBE_TOKEN = "0d66dac5-c096-4499-a9c6-a741a7378956-843339462"

import pdb


def kw2formats(kw, with_first_word=True, is_tagme=False):
        suffix = "_tagme" if is_tagme else ""
        splitted_kw = kw.split()
        formats = {}
        formats["entity"+suffix], formats["words"+suffix] = build_kw_formats(splitted_kw)
        if with_first_word:
            formats["first_entity"+suffix], formats["first_word"+suffix] = build_kw_formats([splitted_kw[0]])
        return formats


def search_entity_matchings(all_kw_formats:dict, wiki2vec):
        detections = []
        for kw in all_kw_formats:
            format_dict = all_kw_formats[kw]
            for format_type in format_dict:
                formatted_keywords = format_dict[format_type]
                found = False
                i = 0
                while (i < len(formatted_keywords)) and (not found):
                    formatted_kw = formatted_keywords[i]
                    kw_embedding = get_vector(wiki2vec, formatted_kw)
                    if kw_embedding is not None:
                        found = True
                        detections.append({"kw":kw,"formatted_kw":formatted_kw,"format":format_type,"embedding":kw_embedding})
                    i += 1
        return detections
    

def ask_user2select_entity(kw, detections, wiki2vec):
    
        start_idx, last_idx, none_idx, write_idx = display_options(kw, detections)
        
        while True:
            chosen_idx = int(input("\nPlease choose an entity ("+str(0)+"-"+str(write_idx)+"): "))
            if not is_valid_idx(chosen_idx, start_idx, last_idx):
                continue
            
            if chosen_idx == write_idx:
                new_kw = input("Please write your own entity: ")
                final_format = new_kw if ("ENTITY" in new_kw) else "ENTITY/"+new_kw
                # final_format = new_kw
                kw_embedding = get_vector(wiki2vec, final_format)
                if kw_embedding is None:
                    print("'"+final_format+"' was not found.")
                    continue
                else:
                    print("Found entity: ", final_format)
                    return kw_embedding, "custom", final_format
            elif chosen_idx == none_idx:
                return None, None, None
            else:
                detection = detections[chosen_idx]
                return detection["embedding"], detection["format"], detection["formatted_kw"]


def select_according_to_priorities(kw, detections, wiki2vec, priorities):
    for priority in priorities:
        for detection in detections:
            if detection["format"] == priority:
                if not (detection["embedding"] is None):
                    print("'"+kw+"'", "-- accepted as", "'"+detection["formatted_kw"]+"'", "("+detection["format"]+")")
                    return detection["embedding"], detection["format"], detection["formatted_kw"]
    return None, None, None


def search_directly(kw, converted_kw, wiki2vec):
    kw_embedding = get_vector(wiki2vec, converted_kw)
    if kw_embedding is None:
        pdb.set_trace()
        Exception("Converted version '"+str(converted_kw)+"' of keyword '"+str(kw)+"' was not found.")
    else:
        return kw_embedding, "forced", converted_kw


def get_wiki2vec_embedding_gensim(kw, wiki2vec, priorities=None, provided_entities={}, use_tagme=False, ask_not_found=False, verbose=True):

    if kw in provided_entities:
        converted_kw = provided_entities[kw]
        print("\nProvided entity:", kw, "-->", converted_kw)
        kw_embedding, accepted_format, formatted_kw = search_directly(kw, converted_kw, wiki2vec)
        return kw_embedding, accepted_format, formatted_kw
    
    # 1st case: kw is already correct format
    formatted_kw = "ENTITY/"+kw
    # formatted_kw = kw
    kw_embedding = get_vector(wiki2vec, formatted_kw)
    if kw_embedding is not None:
        accepted_format = "original"
        return kw_embedding, accepted_format, formatted_kw
    
    print("Original format of keyword "+kw+" does not exist in this wikipedia dump.")
    if ask_not_found:
        not_found = True
        stop=False
        while not_found and not stop:
            replacement_word = input("Please provide replacement word:")
            if replacement_word == "stop":
                stop=True
            else:
                kw_embedding = get_vector(wiki2vec, replacement_word)
                if kw_embedding is not None:
                    print("Found!")
                    accepted_format = "manually_modified"
                    return kw_embedding, accepted_format, replacement_word
    
    # print(kw)
    # pdb.set_trace()
    # 2nd case: build possible formats

    if use_tagme:
        all_kw_formats = {}

        # build several formats for original keyword
        formats = kw2formats(kw, with_first_word=True, is_tagme=False)
        all_kw_formats[kw] = formats.copy()

        # build several formats for keywords detected by tagme
        tagme_annotations_list = run_tagme(kw)
        for annotation in tagme_annotations_list:
            kw_tagme = annotation.entity_title
            tagme_formats = kw2formats(kw_tagme, with_first_word=False, is_tagme=True)
            all_kw_formats[kw_tagme] = tagme_formats.copy()
        
        # search for matchings between formats and wiki2vec
        detections = search_entity_matchings(all_kw_formats, wiki2vec)        
        
        # select best matching
        if priorities is None:
            kw_embedding, accepted_format, formatted_kw = ask_user2select_entity(kw, detections, wiki2vec)
            return kw_embedding, accepted_format, formatted_kw
        else:
            kw_embedding, accepted_format, formatted_kw = select_according_to_priorities(kw, detections, wiki2vec, priorities)
            return kw_embedding, accepted_format, formatted_kw
    else:
        return None, None, None
        


def is_valid_idx(chosen_idx, start_idx, last_idx):
    if not (chosen_idx >= 0 and chosen_idx <= last_idx):
        print("ERROR: chosen entity not in ("+str(start_idx)+"-"+str(last_idx)+").")
        return False
    return True


def display_options(kw, detections):
    none_idx  = len(detections)
    write_idx = none_idx+1
    start_idx = 0
    last_idx  = write_idx
    print("\n### KEYWORD: '"+kw+"' ###")
    print("Formats detected:\n")
    for idx in range(len(detections)):
        detection = detections[idx]
        print("("+str(idx)+"): "+detection["formatted_kw"]+" ("+detection["format"]+")")
    print("("+str(none_idx)+"): Discard keyword.")
    print("("+str(write_idx)+"): Write my own entity.")
    return start_idx, last_idx, none_idx, write_idx


def get_vector(wiki2vec, kw):
    try :
        kw_embedding = wiki2vec.get_vector(kw)
        return kw_embedding
    except:
        return None


def run_tagme(kw, threshold=0.005):
    annotations = tagme.annotate(kw)
    annotations_list = []
    for ann in annotations.get_annotations(threshold):
        annotations_list.append(ann)
    return annotations_list


def build_kw_formats(splitted_kw):
    
    words_formats = []

    words_formats_dict = build_words_formats_dict(splitted_kw)

    for word_format in words_formats_dict:
        list_of_words = words_formats_dict[word_format]
        separation_formats_dict = build_separation_formats_dict(list_of_words)
        words_formats += [separation_formats_dict[mode] for mode in separation_formats_dict]
    
    entity_prefix = "ENTITY/"
    # entity_prefix = ""
    entity_formats = [entity_prefix+word_format for word_format in words_formats]
    
    return entity_formats, words_formats


def build_words_formats_dict(words_list):
    # return dict {"type_of_format":list of words in this format}
    modes_dict = {}
    modes_dict["lower"] = [word.lower() for word in words_list]
    modes_dict["first_letter_upper"] = [first_letter_upper(word) for word in words_list]
    if len(words_list) > 1:
        modes_dict["first_word_first_letter_upper"] = [first_letter_upper(words_list[0])] + [word.lower() for word in words_list[1:]]
    return modes_dict


def first_letter_upper(word):
    return word[0].upper() + word[1:].lower()


def build_separation_formats_dict(words_list):
    space_separation = ' '.join(words_list)
    dash_separation  = '_'.join(words_list)
    return {"space_separation":space_separation, "dash_separation":dash_separation}


def word_formats(word):
    first_letter_upper = word[0].upper() + word[1:].lower()
    lower = word.lower()
    return {"first_letter_upper":first_letter_upper, "lower":lower}


def build_embeddings(kw_dict:dict, embeddings_filename=None, priorities=None, provided_entities={},
                    load_keyedvector=True, keyedvector_file=None, load_entities=False, 
                    load_entities_filename="most_recent", save_entities=False, ask_not_found=False,
                    verbose=True):
    assert isinstance(kw_dict,dict)
    infos = {"formats":{}, "priorities":priorities}
    
    accepted_keywords, rejected_keywords = [], []
    if load_keyedvector:
        assert keyedvector_file is not None
        keyed_vector = KeyedVectors(vector_size=100)
        wiki2vec    = keyed_vector.load(keyedvector_file)
    else:
        assert embeddings_filename is not None
        wiki2vec = KeyedVectors.load_word2vec_format(embeddings_filename, binary=False)
        save_keyedvector = input('Save keyedvector object?')
        if save_keyedvector in {'y', 'Y'}:
            assert keyedvector_file is not None
            wiki2vec.save(keyedvector_file)
    
    if load_entities:
        entities_dict = load_entities_dict(load_entities_filename)
        entities_dict.update(provided_entities)
        provided_entities = entities_dict.copy()
    entities_dict = {}
    save_entities_path = create_new_entities_folder() if save_entities else None
    
    kw_list  = []
    kw2vec   = {}
    for doc in kw_dict:
        for kw in kw_dict[doc]:
            if not(kw in kw_list): # kw not yet processed
                kw_embedding, accepted_format, formatted_kw = get_wiki2vec_embedding_gensim(kw, wiki2vec, priorities=priorities,
                                                                                            provided_entities=provided_entities,
                                                                                            ask_not_found=ask_not_found)
                if kw_embedding is None:
                    print(f'\nKeyword "{kw}" was rejected.')
                    rejected_keywords.append(kw)
                else:
                    accepted_keywords.append(kw)
                    kw2vec[kw] = kw_embedding
                    if save_entities:
                        entities_dict[kw]=formatted_kw
                        save_entities_dict(entities_dict, save_entities_path)
                infos["formats"][accepted_format] = infos["formats"][accepted_format]+1 if (accepted_format in infos["formats"]) else 1
                kw_list.append(kw) # kw has been processed

    infos.update({"nb_kw":len(kw_list), "nb_accepted":len(accepted_keywords), "nb_rejected":len(rejected_keywords)})
    
    print("\nAccepted keywords :\n", accepted_keywords) if verbose else 0
    print("\nRejected keywords :\n", rejected_keywords, "\n") if verbose else 0
    return kw2vec, infos, rejected_keywords


def build_one_hot_encodings(kw_dict, shuffle_config=[True, False], encoding_sizes=["standard", 100, 300, 500]):
    encodings = []
    kw2idx = {}
    kw_idx = 0
    for doc in kw_dict:
        for kw in kw_dict[doc]:
            if kw not in kw2idx:
                kw2idx[kw] = kw_idx
                kw_idx += 1
    eye = np.eye(kw_idx)
    shuffled_eye = eye[np.random.permutation(eye.shape[0])]
    for shuffle in shuffle_config:
        for size in encoding_sizes:
            used_eye = shuffled_eye.copy() if shuffle else eye.copy()
            if size != "standard":
                assert type(size) == int
                if size <= used_eye.shape[1]:
                    Exception("Error: requested encoding size ("+str(size)+") lower than the size of the vocabulary.")
                else:
                    padding = np.zeros((eye.shape[0], size - eye.shape[1]))
                    used_eye = np.concatenate((used_eye,padding),axis=1)
            name = params2names(shuffle, size)
            kw2vec = {}
            for kw in kw2idx:
                idx = kw2idx[kw]
                kw2vec[kw] = used_eye[idx]
            encodings.append({"name":name, "vectors":kw2vec})
    return encodings


def params2names(shuffle, size):
    shuffle_str = "shuffled_" + str(shuffle)
    size_str = "size_" + str(size)
    name = shuffle_str+"-"+size_str
    return name


def build_embedding_folder(data_dir):
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    # time_str = time.strftime("%Hh%Mm%S-%d-%m")
    # results_folder = data_dir + "/" + time_str
    results_folder = data_dir
    
    if not os.path.isdir(results_folder):
        os.mkdir(results_folder)
    return results_folder


def save_one_hot_encodings(one_hot_encodings, data_dir):
    results_folder = build_embedding_folder(data_dir)
    for one_hot_encoding in one_hot_encodings:
        filename = results_folder+"/"+one_hot_encoding["name"] + ".npy"
        results = one_hot_encoding["vectors"]
        np.save(filename, results)


def save_embeddings(kw2embed, infos, data_dir):
    results_folder = build_embedding_folder(data_dir)
    infos_filename = os.path.join(results_folder, 'infos.json')
    utils.save_dict_as_json(data_dict=infos, file_path=infos_filename)
    embeddings_filename = os.path.join(results_folder, 'embeddings.npy')
    np.save(embeddings_filename, kw2embed)


def load_embeddings(filename, verbose=False):
    data=np.load(filename, allow_pickle=True)
    words_embeddings = data[()]
    if verbose:
        for word in words_embeddings:
            embedding = words_embeddings[word]
            print("")
            print(word)
            print(embedding)
    return words_embeddings


def load_entities_dict(filename, verbose=True) -> dict:
    entities_dir = "./saved_entities"
    if filename == "most_recent":
        date_list = [x[0] for x in os.walk(entities_dir)]
        date_list.pop(0)
        date_list = [date.split("saved_entities/")[1] for date in date_list]
        date_list.sort(key=lambda date: datetime.datetime.strptime(date, "%Hh%Mm%S-%d-%m"))
        filename = date_list[-1] + "/" + "entities.json"
    print("*** Loading saved entities file: "+filename) if verbose else 0
    with open(entities_dir+"/"+filename) as json_file:
        entities_dict = json.load(json_file)
    return entities_dict


def create_new_entities_folder():
    entities_dir = "./saved_entities"
    current_date = time.strftime("%Hh%Mm%S-%d-%m")
    new_folder = entities_dir+"/"+current_date
    os.mkdir(new_folder)
    return new_folder


def save_entities_dict(entities_dict, folder):
    filename = folder + "/" + "entities.json"
    with open(filename, "w") as file:
        json.dump(entities_dict, file)