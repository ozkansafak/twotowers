import numpy as np

# Load the function
def evaluate(qb_output, index):
    ranks = []
    hit10 = hit5 = hit3 = 0
    for q in range(len(qb_output)):
        k = 100  # capture 4 nearest neighbors
        D, I = index.search(qb_output[q:q+1], k=k)  # sanity check
        rank = np.argwhere(I[0] == q)
        ranks.append(None if len(rank) == 0 else rank[0,0])
        if len(rank) and rank < 3:
            hit3 += 1
            hit5 += 1
            hit10 += 1
        elif len(rank) and rank < 5:
            hit5 += 1
            hit10 += 1
        elif len(rank) and rank < 10:
            hit10 += 1

    func = lambda x,s: print(f'recall@{str(s):2s} = {x/len(qb_output):.3f}' + '  '*10) 
    # [func(x,s) for x,s in [(hit10,10), (hit5,5), (hit3,3)]];
    func(hit3, 3) 
    
    return hit3 / len(qb_output)


def infer():
    import faiss, numpy as np, pickle, json

    with open('../output/product_info.json', 'r') as file:
        info = json.load(file)

    with open('../output/qb_xb_output.pkl', 'rb') as file:
        (qb_output, xb_output) = pickle.load(file)

    # Train the faiss index (Voronai cells + product quantizer)
    d = qb_output.shape[1]
    nlist = 50  # number of Voronai cells
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist)

    index.train(xb_output)
    index.add(xb_output)  # add vectors to the index
    
    recall3 = evaluate(qb_output, index)

    with open('../output/recall3.json', 'w') as file:
        json.dump({"recall3": recall3}, file)

    return recall3


if __name__ == "__main__":
    infer()






