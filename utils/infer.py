import faiss, numpy as np, pickle, json
from utils.utils_infer import evaluate 

def infer_py_script():
    with open('output/product_info.json', 'r') as file:
        info = json.load(file)
        
    with open('output/qb_xb_output.pkl', 'rb') as file:
        (qb_output, xb_output) = pickle.load(file)
    
    # Train the faiss index (Voronai cells + product quantizer)
    d = qb_output.shape[1]
    nlist = 50  # number of Voronai cells
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist)
    
    index.train(xb_output)
    index.add(xb_output)  # add vectors to the index
    recall3 = evaluate(qb_output, index)
    
    
if __name__ == "__main__":
    return infer_py_script()