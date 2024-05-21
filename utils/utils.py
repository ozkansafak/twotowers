import os
import sys
import ipdb
import pickle
import copy
import json
import logging
import math
import os
import pdb
import random
import string
import subprocess
import time
from collections import defaultdict, namedtuple
from dataclasses import dataclass, field
from functools import cache
from io import BytesIO
from itertools import combinations
from multiprocessing import Pool
from pprint import pprint as _pprint
import numpy as np
from typing import Any, Dict, List, Optional, Set, Tuple
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import pandas as pd
import requests
# import faiss
from openai import OpenAI

from matplotlib.patches import Rectangle
from tqdm import tqdm

OPENAI_API_TIMEOUT = 60
RETRY_COUNT = 3

class OpenAIModel:
    GPT35TURBO = "gpt-3.5-turbo"
    GPT4TURBO = "gpt-4-0125-preview"


pylab.rcParams.update(
    {
        "legend.fontsize": "small",
        "font.size": 12,
        "figure.figsize": (6, 2.2),
        "axes.labelsize": "small",
        "axes.titlesize": "medium",
        "axes.grid": "on",
        "xtick.labelsize": "small",
        "ytick.labelsize": "small",
    }
)

openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"), timeout=OPENAI_API_TIMEOUT, max_retries=RETRY_COUNT
)

# define some constants
hello_prompt = "\nThanks for texting! \nHow can we help you today?\n"
kwargs = {"task": "ProductRequest", "current_cart": {}, "eval": True, "llm_model": None}


def pprint(a, width=130):
    from pprint import pprint

    return pprint(a, width=width)


def print_runtime(start, p_flag=True):
    end = time.time()
    if p_flag:
        print("Runtime: %d min %d sec" % ((end - start) // 60, (end - start) % 60))
        return None
    else:
        return "Runtime: %d min %d sec" % ((end - start) // 60, (end - start) % 60)


def jaccard(set1, set2):
    set1 = set(set1)
    set2 = set(set2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if len(union) > 0 else 1


def get_document(es_document):
    source = es_document["_source"]

    def helper(key: str):
        if key in source:
            return source[key]
        else:
            print(f"{key} is assigned to None")
            return None

    _id = es_document["_id"]
    name = helper("name")
    description = helper("description")
    details = helper("details")
    productType = helper("productType")
    productName = helper("productName")
    categories = helper("categories")
    subCategories = helper("subCategories")

    return {
        "name": name,
        "description": description,
        "details": details,
        "productType": productType,
        "productName": productName,
        "categories": categories,
        "subCategories": subCategories,
        "_id": _id,
    }


def display_single_item(variantId, num_pics=3):
    gs = gridspec.GridSpec(num_pics, 1)
    fig = plt.figure(figsize=(2 * 1, 2 * num_pics))

    url = f"{product_catalog_url}?prodSearch_variantId={variantId}"

    for i, url in enumerate(variant_get_by_id(variantId)["imageUrls"][:num_pics]):
        ax = plt.subplot(gs[i, 0])
        response = requests.get(url)
        if response.status_code != 200:
            print(f"response.status_code:{response.status_code} .. skipping")
            ax.annotate(
                f"({i}, {0})",
                xy=(0.5, 0.5),
                xycoords="axes fraction",
                ha="center",
                va="center",
                fontsize=12,
            )
            continue

        img_buffer = BytesIO(response.content)
        img = mpimg.imread(img_buffer, format="webp")
        ax.imshow(img, aspect="equal")
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    return fig


def generate_utterance(variantId, printer=True):
    # doc is each item in the list `resp['hits']['hits']`.
    # Extract productName, description, details, from `doc`
    start = time.time()

    resp = query_by_variantId(variantId)

    if len(resp["hits"]["hits"]) == 0:
        return None

    doc = resp["hits"]["hits"][0]
    print(f"""{'---'*12}\nvariantId: {doc['_source']['variantId']}""")

    try:
        product_description = (
            f"Here's the product attributes below:"
            + f"""{{'product_name': {doc['_source']['productName']},"""
            + f"""'product_description': {doc['_source']['description']},"""
            + f"""'product_details': {doc['_source']['details']}}}"""
            + f"""Now, generate a single utterance that's to be messaged to the sales representative."""
        )

    except:
        if "productName" not in doc["_source"]:
            print(""" 🙁🙁🙁 SAFAK ERROR: 'productName' not in resp['hits']['hits']['_source'] """)
        if "description" not in doc["_source"]:
            print(""" 🙁🙁🙁 SAFAK ERROR: 'description' not in resp['hits']['hits']['_source'] """)
        if "details" not in doc["_source"]:
            print(""" 🙁🙁🙁 SAFAK ERROR: 'details' not in resp['hits']['hits']['_source'] """)
        print()
        return None

    messages = [
        {
            "role": "system",
            "content": """**SCENARIO:** Imagine a human Customer engaging with a human sales representative from an online department store via a messaging platform. In this scenario, the customer is either the mother or the father in a family of a mom, a dad and two young kids aged between 1 and 10 years.

You're also given the attributes of a specific product in Python dictionary format that the human is seeking to purchase. The sales representative has access to a diverse catalog of one million items, including toys, apparel, educational tools, and more. Your task is to simulate the Customer by crafting a single utterance that would help you ask about the product you want which should match the given product attributes.

**YOUR ROLE:** You're the Customer --you can be the mom or the dad of the family. The family is made of mon, the dad and their two kids aged between 1 and 10 years. The purpose of the Customer is solely to query the product catalog by asking specific questions, NOT broad questions, or gain more info about the products available. Try to describe what you want, then wait for the sales representative to return you a list of products that is relevant to your query.

**OBJECTIVE:** Your goal is to generate a realistic utterance that a shopper might have with the sales associate, aiming to uncover the best products for their family. The purpose is to explore the vast product range with a focus on products suitable for young children (boys or girls) and that caters to the lifestyle of a young family.

**OUTPUT FORMAT:** Plain text. Do NOT wrap it in any kind of quotes.

**RULE:**
- You don't know the details of the items in the product catalog.
- While you can mention product types (e.g. toys, apparel), try also to incorporate attributes about product features, benefits, or usage scenarios relevant to a young family.
- Be creative and think about various shopping scenarios, such as preparing for a birthday, looking for educational tools, or outdoor activities.
- Don't make the product request super specific.
- Make a specific and targeted product request.
- Make the utterances diverse and somewhat different from each other in style.

**EXAMPLE UTTERANCES:**
Here are some example utterances (not for the specific product attributes provided in this conversation but potentially for some other product)
Example 1: "Can you recommend some educational games that would be suitable for a 4-year-old and also fun for the whole family?"
Example 2: "We're planning a family camping trip. What kind of outdoor gear would you recommend for a family with young kids?"
Example 3: "Leo is about to turn 10. Can you provide me some gifts ideas?"
""",
        },
        {"role": "user", "content": product_description},
    ]

    llm_response = openai_client.chat.completions.create(
        model=OpenAIModel.GPT4TURBO,
        temperature=TEMPERATURE,
        messages=messages,
        timeout=OPENAI_API_TIMEOUT,
    )

    utterance = llm_response.choices[0].message.content
    while utterance[0] == utterance[-1] and utterance[0] in ['"', "'"]:
        print(f"utterance[0] == utterance[-1] == {utterance[0]}. Shaving off the ends ")
        utterance = utterance[1:-1]

    print(f"{utterance}\n")

    print_runtime(start)
    return utterance


def get_random_variants(size, plotter=False):
    start = time.time()

    resp = make_random_query(query_random, size=size)

    variantIds, listingIds, names = list(
        zip(
            *[
                [
                    item["_source"]["variantId"],
                    item["_source"]["listingId"],
                    item["_source"]["productName"],
                ]
                for item in resp["hits"]["hits"]
            ]
        )
    )

    print("variantIds: ", variantIds)
    print(f"""len(resp['hits']['hits']) ', {len(resp['hits']['hits'])}""")

    if plotter:
        query = {
            "query": {"bool": {"filter": [{"terms": {"variantId": [item for item in variantIds]}}]}}
        }
        display_image_slate(query, num_pics=3, num_recs=10)

    print_runtime(start)
    return variantIds, listingIds, names


def get_utterances(random_variantIds):
    # Create utterances for each variantId
    start = time.time()

    utterances = []
    for i, variantId in enumerate(random_variantIds):
        utt = generate_utterance(variantId)
        if utt is None:
            utt = "null"
        utterances.append(utt)

    set_nulls = {i for i, utt in enumerate(utterances) if utt == "null"}
    print("set_nulls", set_nulls)
    print(len(utterances) == len(random_variantIds))

    print_runtime(start)
    return utterances, set_nulls


def correct_json(json_string):
    if type(json_string) == float and math.isnan(json_string):
        return "null"
    # json_string = json_string.replace("'", '"')
    json_string = (
        json_string.replace("True", "true").replace("False", "false").replace("None", "null")
    )

    return json_string


def extract_keywords_from_query(args):
    # Designed to tackle a single data point so it can be parallelized
    # OUTPUT: match_keywords, query

    start, i, utt, variantId = args
    pid = os.getpid()
    print(f'{i}.', end='')

    if utt is None:
        print("Invalid utterance")
        return "null", "null"

    docs = [
        Message("outbound", hello_prompt, _id=str(random.randint(0, 1000000))),
        Message("inbound", utt, _id=str(random.randint(0, 1000000))),
    ]
    cnv_obj = Conversation(docs)

    try:
        ret_data = handle_product_request(
            cnv_obj=cnv_obj,
            merchant_id="92",
            vendor="Wizard B2C",
            user_id=214081,
            response_type=None,
            **kwargs,
        )
        query = json.loads(ret_data["eval_info"].es_query)
        for q, item in enumerate(query["query"]["bool"]["should"]):
            if "multi_match" in item:
                match_keywords = item["multi_match"]["query"]
                return match_keywords, json.dumps(query)

        return "null", json.dumps(query)

    except Exception as e:
        print(f"Safak ERROR: {type(e).__name__}")
        return "null", "null"


def apply_multiprocessing(args, func, num_processes=12):
    with Pool(processes=num_processes) as pool:
        results = pool.map(func, args)
    return results


def get_messages(utterance, productName, description, details):
    prompt = f'''
**QUERY PROMPT:** {utterance}
**PRODUCT NAME:** """{productName}"""
**PRODUCT DESCRIPTION:** """{description}"""
**PRODUCT DETAILS:** """{details}"""
**OUTPUT FORMAT:**: You're ONLY going to output "0" or "1" (a single integer and nothing else). Output "1", if you think the product information (i.e. PRODUCT NAME, PRODUCT DESCRIPTION and PRODUCT DETAILS) are relevant, "0" if you think they are NOT relevant.'''

    messages = [
        {
            "role": "system",
            "content": "You are an accurate judge of whether the given product information is relevant to the query prompt.",
        },
        {"role": "user", "content": prompt},
    ]

    return messages


def compute_DCG(scores):
    return sum([score / math.log2(i + 1) for i, score in enumerate(scores, start=1)])


def compute_NDCG(list_scores, K=10):
    list_NDCG = []
    for i, scores in enumerate(list_scores):
        # scores are in (-1, 0, 1). Map them to (0, 1, 2)
        scores = [s for s in scores]
        # print(f"{i:2d}: {scores}")

        IDCG = compute_DCG(sorted(scores, reverse=True)[:K])
        DCG = compute_DCG(scores[:K])
        if IDCG == 0:
            print(f"i:{i} all recs are irrelevant")
            NDCG = 0
        else:
            NDCG = DCG / IDCG

        list_NDCG.append(NDCG)

    return list_NDCG, sum(list_NDCG) / len(list_NDCG)
