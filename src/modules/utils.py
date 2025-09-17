import functools
import json
import logging
import operator
import os
import pickle
import re
import sys
import time
import threading
import requests
from collections import OrderedDict, Counter
from typing import Optional, Union, Literal, Any, List,Dict
import toml
import dspy
from tqdm import tqdm
import httpx
import concurrent.futures
import backoff
from dsp import ERRORS, backoff_hdlr, giveup_hdlr
from langchain_text_splitters import RecursiveCharacterTextSplitter
from trafilatura import extract
from pathlib import Path


class MyOpenAIModel(dspy.OpenAI):
    """A wrapper class for dspy.OpenAI to track token usage."""

    def __init__(
            self,
            model: str = "gpt-3.5-turbo-instruct",
            api_key: Optional[str] = None,
            api_provider: Literal["openai", "azure"] = "openai",
            api_base: Optional[str] = None,
            model_type: Literal["chat", "text"] = None,
            **kwargs
    ):
        super().__init__(model=model, api_key=api_key, api_provider=api_provider, api_base=api_base,
                         model_type=model_type, **kwargs)
        self._token_usage_lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def log_usage(self, response):
        """Log the total tokens from the OpenAI API response."""
        usage_data = response.get('usage')
        if usage_data:
            with self._token_usage_lock:
                self.prompt_tokens += usage_data.get('prompt_tokens', 0)
                self.completion_tokens += usage_data.get('completion_tokens', 0)

    def get_usage_and_reset(self):
        """Get the total tokens used and reset the token usage."""
        usage = {
            self.kwargs.get('model') or self.kwargs.get('engine'):
                {'prompt_tokens': self.prompt_tokens, 'completion_tokens': self.completion_tokens}
        }
        self.prompt_tokens = 0
        self.completion_tokens = 0

        return usage

    def __call__(
            self,
            prompt: str,
            only_completed: bool = True,
            return_sorted: bool = False,
            **kwargs,
    ) -> list[dict[str, Any]]:
        """Copied from dspy/dsp/modules/gpt3.py with the addition of tracking token usage."""

        assert only_completed, "for now"
        assert return_sorted is False, "for now"

        # if kwargs.get("n", 1) > 1:
        #     if self.model_type == "chat":
        #         kwargs = {**kwargs}
        #     else:
        #         kwargs = {**kwargs, "logprobs": 5}

        response = self.request(prompt, **kwargs)

        # Log the token usage from the OpenAI API response.
        self.log_usage(response)

        choices = response["choices"]

        completed_choices = [c for c in choices if c["finish_reason"] != "length"]

        if only_completed and len(completed_choices):
            choices = completed_choices

        completions = [self._get_choice_text(c) for c in choices]
        if return_sorted and kwargs.get("n", 1) > 1:
            scored_completions = []

            for c in choices:
                tokens, logprobs = (
                    c["logprobs"]["tokens"],
                    c["logprobs"]["token_logprobs"],
                )

                if "<|endoftext|>" in tokens:
                    index = tokens.index("<|endoftext|>") + 1
                    tokens, logprobs = tokens[:index], logprobs[:index]

                avglog = sum(logprobs) / len(logprobs)
                scored_completions.append((avglog, self._get_choice_text(c)))

            scored_completions = sorted(scored_completions, reverse=True)
            completions = [c for _, c in scored_completions]

        return completions




class DeepSeekModel(dspy.OpenAI):
    """A wrapper class for DeepSeek API, compatible with dspy.OpenAI."""

    def __init__(
        self,
        model: str = "",
        api_key: Optional[str] = None,
        api_base: str = "",
        **kwargs,
    ):
        super().__init__(model=model, api_key=api_key, api_base=api_base, **kwargs)
        self._token_usage_lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.model = model
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.api_base = api_base
        if not self.api_key:
            raise ValueError(
                "DeepSeek API key must be provided either as an argument or as an environment variable DEEPSEEK_API_KEY"
            )

    def log_usage(self, response):
        """Log the total tokens from the DeepSeek API response."""
        usage_data = response.get("usage")
        if usage_data:
            with self._token_usage_lock:
                self.prompt_tokens += usage_data.get("prompt_tokens", 0)
                self.completion_tokens += usage_data.get("completion_tokens", 0)

    def get_usage_and_reset(self):
        """Get the total tokens used and reset the token usage."""
        usage = {
            self.model: {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
            }
        }
        self.prompt_tokens = 0
        self.completion_tokens = 0
        return usage

    @backoff.on_exception(
        backoff.expo,
        ERRORS,
        max_time=1000,
        on_backoff=backoff_hdlr,
        giveup=giveup_hdlr,
    )
    def _create_completion(self, prompt: str, **kwargs):
        MAX_RETRIES = 3 
        RETRY_DELAY = 2  
        """Create a completion using the DeepSeek API."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            **kwargs,
        }
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(
                    f"{self.api_base}/v1/chat/completions", headers=headers, json=data
                )
                response.raise_for_status()  
                return response.json()  
            except requests.RequestException as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < MAX_RETRIES - 1:
                    print(f"Retrying in {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)  
                else:
                    print("Max retries reached. Skipping...")
                    return None  
                

    def __call__(
        self,
        prompt: str,
        only_completed: bool = True,
        return_sorted: bool = False,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Call the DeepSeek API to generate completions."""
        assert only_completed, "for now"
        assert return_sorted is False, "for now"

        response = self._create_completion(prompt, **kwargs)

        # Log the token usage from the DeepSeek API response.
        self.log_usage(response)

        choices = response["choices"]
        completions = [choice["message"]["content"] for choice in choices]

        history = {
            "prompt": prompt,
            "response": response,
            "kwargs": kwargs,
        }
        self.history.append(history)

        return completions




class LLMConfigs:
    """Configurations for LLM used in different parts of .

    Given that different parts in STORM framework have different complexity, we use different LLM configurations
    to achieve a balance between quality and efficiency. If no specific configuration is provided, we use the default
    setup in the paper.
    """

    def __init__(self):
        
        self.article_gen_lm = None  # LLM used in article generation.
        self.article_polish_lm = None  # LLM used in article polishing.
        
    def init_openai_model(
            self,
            openai_api_key: str,
            openai_type: Literal["openai", "azure"],
            api_base: Optional[str] = None,
            #api_version: Optional[str] = None,
            temperature: Optional[float] = 1.0,
            top_p: Optional[float] = 0.9
    ):

        openai_kwargs = {
            'api_key': openai_api_key,
            'temperature': temperature,
            'top_p': top_p,
            'api_base': api_base,
            #'api_version': api_version,
            
        }
        if openai_type and openai_type == 'azure':
            openai_kwargs['api_base'] = api_base
            # openai_kwargs['api_version'] = api_version
            
            self.article_gen_lm = MyOpenAIModel(model='gpt-4', engine='gpt-4',
                                                **openai_kwargs)
            self.article_polish_lm = MyOpenAIModel(model='gpt-4', engine='gpt-4',
                                                   **openai_kwargs)
            
        elif openai_type and openai_type == 'openai':
            
            
            self.article_gen_lm = MyOpenAIModel(model='gpt-4',
                                                 **openai_kwargs)
            self.article_polish_lm = MyOpenAIModel(model='gpt-4',
                                                    **openai_kwargs)
            
        else:
            logging.warning('No valid OpenAI API provider is provided. Cannot use default LLM configurations.')
        
    def set_article_gen_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.article_gen_lm = model

    def set_article_polish_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.article_polish_lm = model
    
    

    def collect_and_reset_lm_history(self):
        history = [] 
        
        if self.article_gen_lm:
            history.extend(self.article_gen_lm.history)
            self.article_gen_lm.history = []
        if self.article_polish_lm:
            history.extend(self.article_polish_lm.history)
            self.article_polish_lm.history = []

        return history

    def collect_and_reset_lm_usage(self):
        combined_usage = []
        
        if self.article_gen_lm:
            combined_usage.append(self.article_gen_lm.get_usage_and_reset())
        if self.article_polish_lm:
            combined_usage.append(self.article_polish_lm.get_usage_and_reset())
        combined_usage = dict(functools.reduce(operator.add, map(Counter, combined_usage)))

        return combined_usage

    def log(self):
        return OrderedDict(
            {
                
                'article_gen_lm': self.article_gen_lm.kwargs if self.article_gen_lm else None,
                'article_polish_lm': self.article_polish_lm.kwargs if self.article_polish_lm else None,
            }
        )



class BaseCallbackHandler:
    """Base callback handler that can be used to handle callbacks from the STORM pipeline."""


    def on_outline_refinement_end(self, outline: str, **kwargs):
        """Run when the outline refinement finishes."""
        pass

###############################################
# Helper functions for reading and writing files
###############################################


def dump_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def write_str(s, path):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(s)


def load_str(path):
    with open(path, 'r') as f:
        return '\n'.join(f.readlines())


def handle_non_serializable(obj):
    return "non-serializable contents"  # mark the non-serializable part


def load_json(file_name, encoding="utf-8"):

    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'r', encoding=encoding) as fr:
        return json.load(fr)


def dump_json(obj, file_name, encoding="utf-8"):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'w', encoding=encoding) as fw:
        json.dump(obj, fw, default=handle_non_serializable,ensure_ascii=False,indent=4)


###############################################
# Helper functions for post-processing generated text
###############################################




def limit_word_count_preserve_newline(input_string, max_word_count):
    """Limit the word count of a string while preserving complete lines."""

    word_count = 0
    limited_string = ''

    for word in input_string.split('\n'):
        line_words = word.split()
        for lw in line_words:
            if word_count < max_word_count:
                limited_string += lw + ' '
                word_count += 1
            else:
                break
        if word_count >= max_word_count:
            break
        limited_string = limited_string.strip() + '\n'

    return limited_string.strip()




def process_table_of_contents(toc):
    """Convert a table of contents into a tree structure.

    The table of contents is a string with each line representing a heading.
    "#" Title"  indicates section title, "##" Title" to indication subsection title, "###" Title" to indicate subsubsection title, and so on.
    """
    lines = toc.split('\n')

    root = {}
    path = [(root, -1)]

    for line in lines:
        line = line.strip()
        if not line.startswith('#'):
            continue

        # Count only the leading '#' symbols
        level = 0
        for char in line:
            if char == '#':
                level += 1
            else:
                break

        heading = line[level:].strip()
        if len(heading) == 0:
            continue
        while path and path[-1][1] >= level:
            path.pop()

        # Add the new heading
        if path:
            current_dict = path[-1][0]
            current_dict[heading] = {}
            path.append((current_dict[heading], level))

    return root


def convert_outline_into_queries(root):
    
    queries = [] # # 初始化一个空的列表来存放查询结果
    for k in root: # 遍历根节点（目录的顶层）
        queries.extend(convert_outline_into_queries(root[k])) ## 递归调用函数，将子目录的查询结果添加到 queries 列表中
        queries.append(k) 

    return queries


def convert_outline_into_str(root, level):
   
    s = ''
    for k in root:
        s += '#' * level + ' ' + k + '\n'
        s += convert_outline_into_str(root[k], level + 1)

    return s








def clean_incomplete_sentence(text: str) -> str:
    """
    从文本的末尾向前查找，删除最后一部分不完整的内容。
    如果文本最后不是以句号结尾的，删除最后句号前的部分内容。
    如果最后是句号，则不做任何修改。
    """
    
    text = text.strip()

    
    if text.endswith('。'):
        return text

    
    last_period_index = text.rfind('。')

    
    if last_period_index == -1:
        return ''

    
    return text[:last_period_index + 1]


def load_api_key(toml_file_path='../secrets.toml'):
    try:
        with open(toml_file_path, 'r') as file:
            data = toml.load(file)
    except FileNotFoundError:
        print(f"File not found: {toml_file_path}", file=sys.stderr)
        return
    except toml.TomlDecodeError:
        print(f"Error decoding TOML file: {toml_file_path}", file=sys.stderr)
        return
    # Set environment variables
    for key, value in data.items():
        os.environ[key] = str(value)

###############################################
# helper function for web search
###############################################

class WebPageHelper:
    """Helper class to process web pages.

    
    """

    def __init__(
        self,
        min_char_count: int = 100, #考虑的文本字符不能低于150  
        snippet_chunk_size: int = 250, # 
        max_thread_num: int = 10,
    ):
        """
        Args:
            min_char_count: Minimum character count for the article to be considered valid.
            snippet_chunk_size: Maximum character count for each snippet.
            max_thread_num: Maximum number of threads to use for concurrent requests (e.g., downloading webpages).
        """
        self.httpx_client = httpx.Client(verify=False)
        self.min_char_count = min_char_count
        self.max_thread_num = max_thread_num
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=snippet_chunk_size,
            chunk_overlap=0,
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

    def download_webpage(self, url: str):
        try:
            res = self.httpx_client.get(url, timeout=4)
            if res.status_code >= 400:
                res.raise_for_status()
            return res.content
        except httpx.HTTPError as exc:
            print(f"Error while requesting {exc.request.url!r} - {exc!r}")
            return None
    
    def urls_to_articles(self, urls: List[str]) -> Dict:
        """
        Takes a list of URLs as input and returns a dictionary.
        The dictionary maps each URL to the extracted text content of the article.

        Example output:
        {
            "http://example.com/article1": {
                "text": "This is the main content of article 1, with a length greater than 150 characters..."
            },
            "http://example.com/article2": {
                "text": "This is the main content of article 2, also longer than 150 characters..."
            }
        }
        """
        # Use a thread pool to download web pages in parallel.
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_thread_num
        ) as executor:
            htmls = list(executor.map(self.download_webpage, urls))

        articles = {}
        # 解析HTML正文内容
        for h, u in zip(htmls, urls):
            if h is None:
                continue
            article_text = extract(
                h,
                include_tables=False,
                include_comments=False,
                output_format="txt",
            )
            # 过滤短文本
            if article_text is not None and len(article_text) > self.min_char_count:
                articles[u] = {"text": article_text}

        return articles
    
    def urls_to_snippets(self, urls: List[str]) -> Dict:
        # Retrieve the articles corresponding to the URLs
        articles = self.urls_to_articles(urls)
        
        for u in articles:
            articles[u]["snippets"] = self.text_splitter.split_text(articles[u]["text"]) # 增加一个字段的内容
            articles[u]["snippets"] = clean_and_reconstruct_snippets(articles[u]["snippets"])
        
        return articles


###############################################
# Helper functions for outline field extraction
###############################################
def extract_key_elements(data_dict):
    try:
        
        research_theme = data_dict.get('研究主题', '未找到研究主题')
        main_issues = data_dict.get('主要问题', '未找到主要问题')
        analysis_themes = []
        analysis_questions = []
        analysis_conclusions = []
        analysis_angles = []
        analysis_facts = []
        analysis_argument = []

        for section in data_dict.get('分析部分', []):
            theme = section.get('主题', '')
            question = section.get('问题', '')

            conclusion = section.get('结论', '')
            angles = section.get('分析角度', [])
            facts =  section.get('事实',[])
            argument = section.get('论据',[])

            if theme:
                analysis_themes.append(theme)
            if question:
                analysis_questions.append(question)
            if conclusion:
                analysis_conclusions.append(conclusion)
            if angles:
                analysis_angles.append(angles)
            if facts:
                analysis_facts.append(facts)
            if argument:
                analysis_argument.append(argument)

        
        policy_advice = []
        for advice in data_dict.get('政策建议', []):
            content = advice.get('政策建议内容', '')
            target = advice.get('针对的问题或目标', '')
            policy_advice.append({
                "政策建议内容": content,
                "针对的问题或目标": target
            })
        policy_advice_content = []
        policy_advice_target = []
        for advice in data_dict.get('政策建议', []):
            content = advice.get('政策建议内容', '')
            target = advice.get('针对的问题或目标', '')
            if content:
                policy_advice_content.append(content)
            if target:
                policy_advice_target.append(target)

       
        return {
            "研究主题": research_theme,
            "主要问题": main_issues,
            "分析部分主题": analysis_themes,
            "分析部分问题": analysis_questions,
            "分析部分结论": analysis_conclusions,
            "分析部分角度": analysis_angles,
            "分析部分事实":analysis_facts,
            "分析部分论据":analysis_argument,
            "政策建议": policy_advice,
            "政策建议内容":policy_advice_content,
            "政策建议目标":policy_advice_target
        }

    except Exception as e:
        print(f"Error in extract_key_elements: {e}")
        return {}


def clean_and_reconstruct_snippets(snippets, min_line_length=20):
    """
    Clean and reconstruct a list of text snippets.
    
    Parameters:
        snippets (list): List of text snippets, each with a maximum of 100 characters.
        min_line_length (int): Minimum character length of lines to retain, default is 20.
    
    Returns:
        list: Cleaned and reconstructed list of text snippets.
    """
    cleaned_snippets = []
    
    for snippet in snippets:
        # Split snippet by whitespace characters
        lines = re.split(r'\s+', snippet)
        filtered_lines = []
        for line in lines:
            line = line.strip()
            # Retain lines with length greater than or equal to min_line_length
            if len(line) >= min_line_length:
                filtered_lines.append(line)
        # Reconstruct snippet from filtered lines
        if filtered_lines:
            cleaned_snippet = "\n".join(filtered_lines)
            cleaned_snippets.append(cleaned_snippet)
    
    return cleaned_snippets
