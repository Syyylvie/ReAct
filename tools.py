import requests
from recommend import recommend
from langchain.tools import tool
import os
from dotenv import load_dotenv
import re

### find all sentences in a page
def find_sentences(page: str) -> iter:
    pattern = r"[A-Z].*"
    page = re.sub(r'\.\s+', '\n', page)
    matches = re.finditer(pattern, page)
    return matches

load_dotenv()
# """
# (1) search[entity], which returns the first 5 sentences from
# the corresponding entity wiki page if it exists, or else suggests top-5 similar entities from the
# Wikipedia search engine, 
# (2) lookup[string], which would return the next sentence in the page
# containing string, simulating Ctrl+F functionality on the browser. 
# (3) finish[answer], which
# would finish the current task with answer. We note that this action space mostly can only retrieve a
# small part of a passage based on exact passage name, which is significantly weaker than state-of-theart lexical or neural retrievers. The purpose is to simulate how humans would interact with Wikipedia,
# and force models to retrieve via explicit reasoning in language.
# """
# proxies = {"http": os.getenv("HTTP_PROXY"), "https": os.getenv("HTTPS_PROXY")}

# def get_summary(title ):
#     url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
#     ### add User_Agent in headers
#     headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36"}
#     response = requests.get(url, headers=headers, )
#     if response.status_code == 200:
#         data = response.json()
#         extract = data["extract"]
#         print(extract)
#     else:
#         print(f"页面不存在或API请求失败: {response.status_code}")

# get_summary("einstein")

headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36"}

### load wordvecs
@tool
def search_entity(entity: str) -> (str, iter):
    """returns the first 5 sentences from the corresponding entity wiki page if it exists, 
    or else suggests top-5 similar entities from the
    Wikipedia search engine, 
    """
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{entity}"
    response = requests.get(url, headers=headers, )
    ### if page does not exist
    if response.status_code == 400:
        ### return candidates 
        candidates = recommend(entity)
        return f"Couldn't find {entity}. Similar: {candidates}"
    ### extract page
    page = response.json()["extract"]
    ### get sentence iterator
    matches = find_sentences(page)
    ### return first 5 sentences and current str pointer
    res = ""
    for i, match in enumerate(matches):
        res += match.group()
        if i == 5:
            return res, matches
    return res, matches
    
@tool
def find_next(entity: str, matches: iter) -> (str, iter):
    """
    if search_entity returns 5 sentences irrelavent to the entity or the results
    cannot invoke the answer, 
    find next sentence containing keyword from the current page
    used when search_entity() successfully returns 5 sentences
    """
    if matches is None:
        print("No next sentence")
        return (None, None)
    for match in matches:
        sentence = match.group()
        if entity in sentence:
            return (sentence, matches)
    print(f"No next sentence containing {entity}")
    return (None, None)

