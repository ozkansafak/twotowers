from utils.utils import *
import torch
torch.manual_seed(0)
import random
random.seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class TwoTowerNetwork(nn.Module):
    def __init__(self, d, hidden_dim, learning_rate=0.0001):
        super(TwoTowerNetwork, self).__init__()

        # qb_tower
        self.qb_tower = nn.Sequential(
            nn.Linear(d, hidden_dim),
            nn.ReLU(),
            nn.Linear(d, hidden_dim),
        )

        # xb_tower 
        self.xb_tower = nn.Sequential(
            nn.Linear(d, hidden_dim),
            nn.ReLU(),
            nn.Linear(d, hidden_dim),         
        )

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.train_loss = [None]
        self.epoch = 0
        self.epochs = [0]
        
    def forward(self, u, v, batch_size=1024, label_smoothing=0.1):
        qb_output = self.qb_tower(u)
        xb_output = self.xb_tower(v)

        targets_batch = torch.arange(batch_size, dtype=torch.long)

        logits = dot_product = torch.einsum ('ik, jk -> ij', qb_output, xb_output)
        probs = F.softmax(logits, dim=-1)
        loss = F.cross_entropy(logits, targets_batch, label_smoothing=label_smoothing)

        return probs, loss, qb_output, xb_output

    def inference(self, u, v):
        u_output = self.qb_tower(u)
        v_output = self.xb_tower(v)

        return u_output, v_output

    def size(self):
        total = 0
        for name, params in self.named_parameters():
            a = params.shape
            total += a[0] * (1 if len(a) == 1 else a[1]) 
            print(f'{str(tuple(a)):15s}', total)

    def train(self, qb_train, xb_train, num_epochs, batch_size):
        n =  len(qb_train) // batch_size
        
        for e in range(num_epochs):
            for start in range(0, len(qb_train), batch_size):
                if start + batch_size > len(qb_train):
                    continue
                qb_batch = qb_train[start:start+batch_size]
                xb_batch = xb_train[start:start+batch_size]

                # Forward pass: compute predictions
                logits, loss, _, _ = self(qb_batch, xb_batch, batch_size)

                # Backward pass and optimization
                self.optimizer.zero_grad()  # zero out the gradient ops
                loss.backward()  # compute gradients
                self.optimizer.step()  # apply gradients on the parameters.
                self.train_loss.append(loss.item())
                
                self.epochs.append(self.epochs[-1] + 1/n)
            self.epoch += 1
            print(f"Epoch [{self.epoch:2d}], Loss: {loss.item():.4f}", end='\r')

    def plot(self, qb_train, batch_size):
        #train_epochs = np.linspace(0, len(self.train_loss) / (len(qb_train) // batch_size), len(self.train_loss))
        plt.plot(self.epochs, self.train_loss, 'k', alpha=.7)
        plt.title('Train loss')
        plt.xlabel('number of epochs');        


def generate_gpt_queries(name, details, description, size=3):
    messages = [
        {
            "role": "system",
            "content": "You are playing the role of a parent who has one or more children between the ages 2-12. You are usually busy and overwhelmed with errands you need to run. You're submitting a text message on a web app text messaging platform to provide a product request to a sales representative. Your texting mannerism is casual as you don't have too much time to spare for your shopping experience.",
        },
        {
            "role": "user",
            "content": f"""**INPUT:** Here's the name, description and details of a product you need to purchase:
"Product Name": '''{name}'''
"Product Details": '''{details}'''
"Product Description": '''{description}'''

**STYLE:** Provide a product request in plain English, as if a customer is conversing with a sales agent on the phone explaining your need. You could also provide the context for who you need it for.

**OUTPUT:** Provide one single utterance in a string format"""
        },
    ]
    for trial in range(3):
        try:
            llm_response = openai_client.chat.completions.create(
                model=OpenAIModel.GPT35TURBO,
                temperature=TEMPERATURE,
                messages=messages,
                timeout=OPENAI_API_TIMEOUT,
            )

            # queries = list(json.loads(llm_response.choices[0].message.content).values())
            queries = llm_response.choices[0].message.content

        except Exception as e:
            print(f'Exception in get_queries {e}. llm_response.choices[0].message.content:{llm_response.choices[0].message.content}. try again.')

    return queries


def shuffle_and_split(qb, xb, split=0.8):
    np.random.seed(41)  # seed(36): train for 200 epochs, recall@3=.37
    idx = np.arange(len(qb))
    np.random.shuffle(idx)
    xb = xb[idx]
    qb = qb[idx]

    n = len(idx)
    idx_train, idx_test = idx[:int(n*split)], idx[int(n*split):]
    xb_train, xb_test = xb[:int(n*split)], xb[int(n*split):]
    qb_train, qb_test = qb[:int(n*split)], qb[int(n*split):]

    return (qb_train, qb_test), (xb_train, xb_test), (idx_train, idx_test)


def write_raw_queries(queries, query_embeddings, product_embeddings):
    start = time.time()
    with open('output/raw_queries.pkl', 'wb') as file:
        pickle.dump ((queries, query_embeddings, product_embeddings), file)
    print(f"written to output/raw_queries.pkl: {print_runtime(start, False)}")


def load_raw_queries():
    with open('output/raw_queries.pkl', 'rb') as file:
        queries, query_embeddings, product_embeddings = pickle.load(file)
    print('loaded (queries, query_embeddings, product_embeddings) from  output/raw_queries.pkl')
    return queries, query_embeddings, product_embeddings


def get_text_embeddings(scroll_id, elSearch, scroll_time="1m"):
    start = time.time()
    _ids = []
    product_embeddings = dict()
    queries = dict()
    query_embeddings = dict()

    for i in range(30):  # number of calls within the scroll context
        print(f'i:{i} ', end='')
        response = elSearch.client.scroll(scroll_id=scroll_id, scroll=scroll_time)
        scroll_id = response["_scroll_id"]

        for product in response['hits']['hits']:  # size = 10
            name, details, description = get_product_info(product)
            _id = product['_id']
            _ids.append(_id)

            sentence = f"""Product Name:"{name}", Product Details:"{details}", Product Description:"{description}" """
            product_embeddings[_id] = sbert.encode(sentence)
            queries[_id] = get_queries(name, details, description)
            query_embeddings[_id] = sbert.encode(queries[_id])

            print('.', end='')
        print(f" {print_runtime(start, False)}")

    # Close the scroll
    elSearch.client.clear_scroll(scroll_id=scroll_id)
    return queries, query_embeddings, product_embeddings, _ids


def write_output_embeddings(model, qb_test, xb_test):
    qb_output, xb_output = model.inference(qb_test, xb_test)
    
    # Cast qb and xb as numpy arrays
    qb_output = qb_output.detach().numpy()
    xb_output = xb_output.detach().numpy()
    
    with open('output/qb_xb_output.pkl', 'wb') as file:
        pickle.dump((qb_output, xb_output), file)
