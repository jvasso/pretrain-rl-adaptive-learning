import pdb
import embedding_functions

# params
corpus_name = "corpus2"
date = "standard1"

corpus_data_dir = "./corpus_data"
filename = "embeddings.npy"
final_filename = corpus_data_dir+"/"+corpus_name+"/"+"embeddings"+"/"+date+"/"+filename

words_encodings = embedding_functions.load_embeddings(filename=final_filename, verbose=True)
pdb.set_trace()