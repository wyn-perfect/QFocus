
import os
from typing import Callable, Union, List
import json

import dspy

from dsp import backoff_hdlr, giveup_hdlr
from modules.utils import WebPageHelper,clean_and_reconstruct_snippets
from langchain_text_splitters import RecursiveCharacterTextSplitter


def save_results_to_json(
    results,
    json_dir: str = "search_results_db",  # Directory path for the JSON file
    json_file: str = "search_data.json"  # File name for the JSON file
):
    """
    Save search results to a global JSON file, overwriting the old file each time.

    Args:
        results: List of search results, in the same structure as TavilySearchRM output.
        json_dir: Path to the global file directory (default: STORM project-specific data directory).
        json_file: JSON file name (default: search_data.json).
    """
    # Construct the full file path
    json_path = os.path.join(json_dir, json_file)
    
    try:
        # Create the directory if it doesn't exist (exist_ok=True handles existing directories)
        os.makedirs(json_dir, exist_ok=True)

        # Write the new search results to the JSON file, overwriting any old data
        with open(json_path, 'w', encoding='utf-8-sig') as f:
            json.dump(
                results,  # Write the new search results
                f,
                ensure_ascii=False,
                indent=2,  # Compact indentation for readability
                separators=(',', ':')  # Remove unnecessary spaces
            )

        print(f"Successfully updated: {len(results)} search results written")

    except Exception as e:
        print(f"Operation failed: {str(e)}")


import time

class TavilySearchRM(dspy.Retrieve):
    """Retrieve information from custom queries using Tavily."""
    def __init__(
        self,
        tavily_search_api_key="xxx",
        k: int = 3, 
        is_valid_source: Callable = None,
        min_char_count: int = 100,
        snippet_chunk_size: int = 250, 
        webpage_helper_max_threads=10,
        include_raw_content=False ,
        max_retries: int = 3, 
        retry_delay: int = 5   
    ):
        super().__init__(k=k)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=snippet_chunk_size,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False,
            separators=[
                "\n\n",
                "\n",
                ".",
                "\uff0e",  # Fullwidth full stop
                "\u3002",  # Ideographic full stop
                ",",
                "\uff0c",  # Fullwidth comma
                "\u3001",  # Ideographic comma
                " ",
                "\u200B",  # Zero-width space
                "",
            ],
        )
        try:
            from tavily import TavilyClient
        except ImportError as err:
            raise ImportError("Tavily requires `pip install tavily-python`.") from err

        if not tavily_search_api_key and not os.environ.get("TAVILY_API_KEY"):
            raise RuntimeError(
                "You must supply tavily_search_api_key or set environment variable TAVILY_API_KEY"
            )
        elif tavily_search_api_key:
            self.tavily_search_api_key = tavily_search_api_key
        else:
            self.tavily_search_api_key = os.environ["TAVILY_API_KEY"]

        self.k = k
        self.webpage_helper = WebPageHelper(
            min_char_count=min_char_count,
            snippet_chunk_size=snippet_chunk_size,
            max_thread_num=webpage_helper_max_threads,
        )

        self.usage = 0
        self.max_retries = max_retries  # Maximum retries count
        self.retry_delay = retry_delay  # Delay between retries

        # Creates client instance that will use search. Full search params are here:
        self.tavily_client = TavilyClient(api_key=self.tavily_search_api_key)
        self.include_raw_content = include_raw_content

        if is_valid_source:
            self.is_valid_source = is_valid_source
        else:
            self.is_valid_source = lambda x: True

    def get_usage_and_reset(self):
        usage = self.usage
        self.usage = 0
        return {"TavilySearchRM": usage}

    def forward(
        self, query_or_queries: Union[str, List[str]], exclude_urls: List[str] = []
    ):
        """Search with TavilySearch for self.k top passages for query or queries
        Args:
            query_or_queries (Union[str, List[str]]): The query or queries to search for.
            exclude_urls (List[str]): A list of urls to exclude from the search results.
        Returns:
            a list of Dicts, each dict has keys of 'description', 'snippets' (list of strings), 'title', 'url'
        """
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        self.usage += len(queries)

        collected_results = []
        combind_results = []
        
        total_queries = len(queries)
        successful_queries = 0 
        failed_queries = 0 
        total_successful_links = 0  
        total_faith_links = 0  

        
        visited_urls = set()
        
        for query in queries:
            args = {
                "query": query,         
                "max_results": self.k,  
                "include_raw_content": self.include_raw_content,
            }
            
            retries = 0
            query_successful = False  
            successful_links_in_query = 0  
            faithful_links_in_query = 0  
            while retries < self.max_retries:
                try:
                    # 尝试请求 API
                    responseData = self.tavily_client.search(**args)
                    results = responseData.get("results",[]) 
                    if not results:
                        print(f"No results found for query: {query}")
                        break  
                    
                    
                    # 提取所有 URL 并批量处理
                    urls = []
                    # urls = [d.get("url") for d in results if d.get("url")]
                    for d in results:
                        url = d.get("url")
                        if url and url not in visited_urls and url not in exclude_urls:  # 检查 URL 是否已出现过
                            visited_urls.add(url)  
                            urls.append(url) 

                    webpage_articles = self.webpage_helper.urls_to_articles(urls)
                    webpage_snippets = self.webpage_helper.urls_to_snippets(urls)
                    
                    
                    combined_result = {
                        "articles": webpage_articles,
                        "snippets": webpage_snippets
                    }
                    combind_results.append(combined_result)
                    


                    for d in results:
                        if not isinstance(d, dict):
                            print(f"Invalid result: {d}\n")
                            continue
                        try:
                            # ensure keys are present
                            url = d.get("url", None)
                            title = d.get("title", None)
                            description = d.get("content", None)
                            snippets = []

                            if d.get("raw_content"):#The cleaned and parsed HTML content of the search result. 
                                snippets = self.text_splitter.split_text(d.get("raw_content"))
                                # 清理一下snippets
                                snippets = clean_and_reconstruct_snippets(snippets)

                            else:
                                snippets.append(d.get("content"))
                                snippets = clean_and_reconstruct_snippets(snippets)

                            if not all([url, title, description, snippets]):
                                raise ValueError(f"Missing key(s) in result: {d}")
                            
                            if url in webpage_articles:
                                webpage_article = webpage_articles[url].get("text", "")
                                webpage_snippets_list = webpage_snippets[url].get("snippets", [])

                                if not webpage_snippets_list:
                                    webpage_snippets_list = [description]
                            else:
                                webpage_article = [description]
                                webpage_snippets_list = snippets
                            
                            if self.is_valid_source(url) and url not in exclude_urls:
                                result = {
                                    "query":query,
                                    "url": url,
                                    "title": title,
                                    "description": description,
                                    "snippets": snippets,
                                    "webpage_article": webpage_article,
                                    "webpage_snippets": webpage_snippets_list,
                                }
                                collected_results.append(result)
                                successful_links_in_query += 1  # 当前查询中成功处理的链接数加1

                        except Exception as e:
                            faithful_links_in_query+=1 # 失败的链接+！
                            print(f"Error occurs when processing result: {e}\n")
                            print(f"Error occurs when searching query: {query}")
                    query_successful = True 
                    successful_queries += 1  
                    total_successful_links += successful_links_in_query  
                    total_faith_links+=faithful_links_in_query
                
                    break  
                except Exception as e:
                    retries += 1
                    print(f"Error occurred, attempt {retries}/{self.max_retries}: {e}")
                    if retries < self.max_retries:
                        print(f"Retrying in {self.retry_delay} seconds...")
                        time.sleep(self.retry_delay)  
                    else:
                        failed_queries += 1  
                        print(f"Max retries reached. Skipping this query: {query}")
                        break  
        
        save_results_to_json(collected_results)
        save_results_to_json(combind_results,json_file="combind.json")
       
        
        print(f"Total queries: {total_queries}")
        print(f"Successful queries: {successful_queries}")
        print(f"Failed queries: {failed_queries}")
        print(f"Total successful links processed: {total_successful_links}")
        print(f"Total failed links processed: {total_faith_links}")
        return collected_results


