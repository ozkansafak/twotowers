from utils.utils_imports import *

class TwoTowerNetwork(nn.Module):
    def __init__(self, d):
        super(TwoTowerNetwork, self).__init__()

        # qb_tower
        self.qb_tower = nn.Sequential(
            nn.Linear(d, d, bias=False),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(d, d, bias=False),
        )

        # xb_tower 
        self.xb_tower = nn.Sequential(
            nn.Linear(d, d, bias=False),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(d, d, bias=False),
        )

        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(2 * d, 1, bias=False),
            nn.Dropout(config['dropout']),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=config['learning_rate'])
        self.train_loss = [None]
        self.epochs = [0]

        # print model details
        self.print_deets()

    def forward(self, qb_batch, xb_batch, targets_batch, sample_weights):    
        """
            qb_batch.shape = (b, b, d)
            xb_batch.shape = (b, b, d)
            targets_batch.shape = (b, b)
        """

        qb_output = self.qb_tower(qb_batch)
        xb_output = self.xb_tower(xb_batch)
        out = torch.concatenate((qb_output, xb_output), axis=-1) # shape: (b, b, 2 * d)

        logits = self.mlp(out).squeeze(-1) # shape: (b, b)
        probs = F.sigmoid(logits)
        loss = F.binary_cross_entropy_with_logits(logits, targets_batch, weight=sample_weights)

        return probs, loss, qb_output, xb_output

    def fit(self, qb_train, xb_train):
        self.train()
        n = (len(qb_train) // config['batch_size']) * config['batch_size'] # 20480
        print(f"model.fit n:{n}, num_batches:{len(qb_train) // config['batch_size']}, qb_train.shape: {qb_train.shape}")

        for b in range(0, n, config['batch_size']):
            qb_seg = qb_train[b:b+config['batch_size']] # shape: (b, d) 
            xb_seg = xb_train[b:b+config['batch_size']] # shape: (b, d) 

            qb_batch, xb_batch, targets_batch, sample_weights = fused_trainset(qb_seg, xb_seg)

            start = time.time()
            # Forward pass: compute predictions
            logits, loss, _, _ = self.forward(qb_batch, xb_batch, targets_batch, sample_weights)

            # Backward pass and optimization
            self.optimizer.zero_grad()  # zero out the gradient ops
            loss.backward()  # compute gradients
            self.optimizer.step()  # apply gradients on the parameters.
            self.train_loss.append(loss.item())

            self.epochs.append(round(self.epochs[-1] + config['batch_size'] / n, 6)) 
            print(f"epoch {self.epochs[-1]:.3f}, train_loss: {self.train_loss[-1]:.5f}")

        print(f"Epoch [{self.epochs[-1]:.2f}], Loss: {loss.item():.4f}")

    def inference(self, u, v):
        self.eval()
        u_output = self.qb_tower(u)
        v_output = self.xb_tower(v)
        out = torch.concatenate((u_output, v_output), axis=-1) # shape: (b, b, 2 * d)
        logits = self.mlp(out).squeeze(-1) # shape: (b, b)
        probs = F.softmax(logits, dim=1)

        return u_output, v_output

    def print_deets(self):
        [print(f"{key:16s}: {val}") for key, val in config.items()]
        print()
        print(self)
        self.print_size()

    def print_size(self):
        num_params = 0
        for name, params in self.named_parameters():
            a = params.shape
            num_params += a[0] * (1 if len(a) == 1 else a[1]) 

        print(f"num_params:{num_params/1e6:.2f} million ")

    def plot(self, qb_train, list_test_epochs, list_recall3):
        plt.figure(figsize=(10, 2.2))
        ax1 = plt.subplot(1,2,1)
        ax1.plot(self.epochs, self.train_loss, 'k', alpha=.7)
        ax1.set_title('Train loss')
        ax1.set_xlabel('number of epochs');

        ax2 = plt.subplot(1,2,2)
        ax2.plot(list_test_epochs, list_recall3, 'k', alpha=.7)
        ax2.set_title(f'recall@3: {max(list_recall3):.3f}')
        ax2.set_xlabel('number of epochs')
        [(ax.set_xlim(0,len(list_test_epochs)), ax.set_ylim(0)) for ax in [ax1, ax2]]
        ax2.set_ylim(0,1)
        plt.show()


def fused_trainset(qb_seg, xb_seg):
    """ 
        qb_seg is one batch of queries. 
        xb_seg is one batch of items.
        (qb_seg[i], xb_seg[i]), i = 0, ... b-1, makes a postive query-item pair 
        qb_seg.shape = (b, d)
        xb_seg.shape = (b, d)
    """
    start = time.time()
    n = len(qb_seg) # n: batch_size
    qb_batch = []
    xb_batch = []
    targets_batch = []
    sample_weights = []

    for i in range(n):
        for q in range(n):
            qb_batch.append(qb_seg[i])
            xb_batch.append(xb_seg[q])
            if i == q:
                targets_batch.append(1.0)
                sample_weights.append(1.0)
            else:
                targets_batch.append(0.0)
                sample_weights.append(1.0)
            
    qb_batch = torch.stack(qb_batch)
    xb_batch = torch.stack(xb_batch)
    targets_batch = torch.tensor(targets_batch)
    sample_weights = torch.tensor(sample_weights)

    print(f"fused_trainset: {tuple(qb_batch.shape)} {tuple(targets_batch.shape)} {time.time()-start:.2f} sec -> ", end='')

    # qb_batch.shape == (b * b, d)
    return qb_batch, xb_batch, targets_batch, sample_weights


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


def shuffle_and_split(qb, xb, split=0.8, seed=None):
    seed = seed or np.random.choice((2**32 - 1))
    np.random.seed(seed)  # seed(36): train for 200 epochs, recall@3=.37
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

    # Pass the input vectors through the Two-Tower Network
    qb_output, xb_output = model.inference(qb_test, xb_test)

    # Cast qb_output and xb_output as numpy arrays
    qb_output = qb_output.detach().numpy()
    xb_output = xb_output.detach().numpy()

    # Write to output vectors to pickle file
    with open('output/qb_xb_output.pkl', 'wb') as file:
        pickle.dump((qb_output, xb_output), file)

