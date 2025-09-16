import concurrent.futures # 并发执行：多线程处理搜索任务
import functools  # 装饰器工具：记录函数执行时长
import json  # JSON数据处理：保存对话日志 
import logging  # 日志系统：记录运行状态    
import os  # 文件路径处理  
import time   # 耗时统计 
from typing import List, Optional     
from concurrent.futures import as_completed  # 获取线程执行结果
from dataclasses import dataclass, field  # 数据类定义：参数容器
from modules.google_reference import SearchWithExclusions_no_focus, SearchWithExclusions_with_focus,DetailedQueryGenerate_no_focus,DetailedQueryGenerate_with_focus # 引入用于检索的类
from modules.rm import TavilySearchRM  # 导入检索模块
from modules.reference import extract_elements_from_evident,extract_elements_from_frame
from sentence_transformers import SentenceTransformer
import concurrent.futures
from modules.frame import article_generate_frame_batch,article_generate_frame_one
from modules.write_page import (SearchCollectedInfo,
                     Section_Write_no_focus,
                     Section_Write_with_focus,
                     PolishArticleModule,
                     
                     )
from modules.utils import (LLMConfigs, dump_json, write_str, process_table_of_contents,
                           load_json, load_str, extract_key_elements, clean_incomplete_sentence,BaseCallbackHandler)

logging.basicConfig(level=logging.INFO, format='%(name)s : %(levelname)-8s : %(message)s')
logger = logging.getLogger(__name__)

def log_execution_time(func):
    """Decorator to log the execution time of a function."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"{func.__name__} executed in {execution_time:.4f} seconds")
        return result

    return wrapper


@dataclass
class DeepSearchRunnerArguments:
    """Arguments for controlling our pipeline."""
    
    output_dir: str = field(
        metadata={"help": "Output directory for the results."},
    )

    input_dir: str = field(
        metadata={"help": "input directory "},
    )
    base_dir: str = field(
        default='result-new',
        metadata={"help": "base directory "},
    )
    
    search_top_k: int = field(
        default=3,
        metadata={"help": "Top k search results to consider for each search query."},
    )
    retrieve_top_k: int = field(
        default=3,
        metadata={"help": "Top k collected references for each section title."},
    )
    with_focus: bool = field(
        default=False,
        metadata={"help": "Whether to use focus-based search queries."},
    )




class DeepSearchRunner:
    
    def __init__(self,
                 args: DeepSearchRunnerArguments,
                 llm_configs: LLMConfigs):
        self.args = args
        self.llm_configs = llm_configs
        # Caveat: DeepSearchRunner does not support multi-threading.
        self.article_dir_name = ""

    @staticmethod
    def convert_search_result_to_url_to_info(search_results):
        url_to_info = {}

        for result in search_results:
            url = result['url']
            if url in url_to_info:
                url_to_info[url]['snippets'].extend(result['snippets'])
                url_to_info[url]['snippets'] = list(set(url_to_info[url]['snippets']))
            else:
                url_to_info[url] = result

        return url_to_info


    @log_execution_time
    def _generate_ref(self,
                      search_top_k :int = 2, # default = 2 
                      json_path: str = '/Users/wangyini/Desktop/code/storm-branch/storm/dataset/articles_new.json',
                      input_dir: str ="/Users/wangyini/Desktop/code/storm-branch/storm/results/share",
                      tavily_api_keys: List[str] = [], #
                      start_index: int = 0,  # 指定起始文章序号
                      end_index: Optional[int] = None  # 指定结束文章序号
                      ):
        """
        the function is to extract the fact and evident from initial artical,and then generate the search query,
        and generate the initial relative article 

        Args:
            title : through title to find the content of the article 
            json path : the position of article
            serch_top_k: the top k result of search
        Returns:
                url_to_info:
                {
                    url1: {'url': str, 'title': str, 'snippets': List[str]},
                    ...
                }
        """

        # 初始化当前使用的 API Key 索引和文章计数器
        current_key_index = 0
        
        # Load articles from the provided JSON file
        articles = load_json(json_path)
        start_index = 0
        end_index = 1
        articles = articles[start_index:end_index]
        
        # Process articles starting from start_index
        for index, article in enumerate(articles, start=start_index):
            
            title = article.get("title","")
            background = article.get("fact")
            ground_truth_url = article.get("url") # exclude url
            
            # Initialize search engines
            tavily_search = TavilySearchRM(k=search_top_k,tavily_search_api_key= tavily_api_keys[current_key_index])

            search_with_exclusions_no_focus = SearchWithExclusions_no_focus(
                engine=self.llm_configs.article_gen_lm,
                api_key=tavily_api_keys[current_key_index],  
                search_top_k=self.args.retrieve_top_k
            )
            search_with_exclusions_with_focus = SearchWithExclusions_with_focus(
                engine=self.llm_configs.article_gen_lm,
                api_key=tavily_api_keys[current_key_index],
                search_top_k=self.args.retrieve_top_k
            )
            

            # Define file paths for extracting evidence and frame
            evident_file_path = os.path.join(input_dir, "arc", "evident", f"{title}_evident.json")
            frame_file_path = os.path.join(input_dir, "arc", "frame", f"{title}_frame.json")

            # Extract evidence and frame data
            extracted_evident = extract_elements_from_evident(evident_file_path)
            extracted_frame = extract_elements_from_frame(frame_file_path)

            # Generate initial search query from extracted evidence
            init_query = list(set(extracted_evident["事实"]
                                + extracted_evident["论据"]))
            
            searched_results = tavily_search(list(set(init_query)), exclude_urls=[ground_truth_url])
            url_to_info = DeepSearchRunner.convert_search_result_to_url_to_info(searched_results)
            
            
            # according to evident to collect reference 
            file_path = os.path.join(self.args.output_dir, "tmp","evident_ref", f"ref_{title}_evident.json")
            dump_json(url_to_info, file_path)
            print(f"Saved initial search results to {file_path}")
            

            # Extract necessary information from the frame
            
            research_topic = extracted_frame.get("研究主题", "")
            themes = extracted_frame.get("主题", [])
            questions = extracted_frame.get("问题", [])
            analysis_angles = extracted_frame.get("分析角度", [])

            # Define the search query generator function
            def generate_search_queries(theme: str, question: str, analysis_angle: List[str]):

                # no focus(angle) 的初始查询
                search_result_no_focus = search_with_exclusions_no_focus.forward(
                    topic=research_topic,
                    background=background,
                    section_theme=theme,
                    ground_truth_url=ground_truth_url
                )
                url_to_info_no_focus = DeepSearchRunner.convert_search_result_to_url_to_info(search_result_no_focus.searched_results)
                queries_no_focus = search_result_no_focus.queries

                # Save initial search results
                init_info = {
                    "queries": queries_no_focus,
                    "url_to_info": url_to_info_no_focus
                }

                #  with focus(angle)的初始查询
                search_result_with_focus = search_with_exclusions_with_focus.forward(
                    topic=research_topic,               # The topic of the article
                    question=question,         # The specific question the chapter addresses
                    section_theme=theme,     # The section title as the perspective
                    sub_dimensions=analysis_angle,  # The key sub-dimensions the chapter focuses on
                    background_info=background,  # Background information about the topic
                    ground_truth_url=ground_truth_url  # The URL to exclude from search results
                )
                url_to_info_with_focus = DeepSearchRunner.convert_search_result_to_url_to_info(search_result_with_focus.searched_results)
                queries_with_focus = search_result_with_focus.queries
                init_deep_info = {
                    "queries": queries_with_focus,
                    "url_to_info": url_to_info_with_focus
                }

                return init_info,init_deep_info
                

            # Using concurrent futures to parallelize the generation of queru 
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                future_to_theme = {}
                info_with_angle ={}
                info_no_angle ={}
            
                # Submit tasks for each section
                for theme, question, analysis_angle in zip(themes, questions, analysis_angles):
                    future = executor.submit(generate_search_queries, theme, question, analysis_angle) # 这个是每个任务执行结果返回的对象叫future
                    future_to_theme[future] = theme #future_to_theme 字典 键值是future  值是当前的主题
                    
                for future in concurrent.futures.as_completed(future_to_theme):
                    init_info, init_deep_info = future.result() # 提取每个任务的结果
                    theme = future_to_theme[future]
                    info_no_angle[theme] = init_info
                    info_with_angle[theme] = init_deep_info

            # Process and save results for no_angle
            #Init refers to the reference  obtained without using evident "while 'no angle' represents the reference found by generating queries based solely on the question."
            url_to_info_init_no_angle = {}
            for theme, info in info_no_angle.items():
                url_to_info_init_no_angle.update(info["url_to_info"])
            file_path = os.path.join(self.args.output_dir, "ref", "no_angle", "no_angle_init", f"ref_{title}.json")
            dump_json(url_to_info_init_no_angle, file_path)
            print(f"Saved no_angle initial results to {file_path}")


            # Combine with initial url_to_info
            url_to_info_no_angle_full = {**url_to_info_init_no_angle,**url_to_info}
            file_path = os.path.join(self.args.output_dir, "ref", "no_angle", "no_angle_full", f"ref_{title}.json")
            dump_json(url_to_info_no_angle_full, file_path)
            print(f"Saved no_angle full results to {file_path}")

            # Process and save results for with angle queries
            url_to_info_init_with_angle = {}
            for theme, info in info_with_angle.items():
                url_to_info_init_with_angle.update(info["url_to_info"])
            file_path = os.path.join(self.args.output_dir, "ref", "with_angle", "with_angle_init", f"ref_{title}.json")
            dump_json(url_to_info_init_with_angle, file_path)
            print(f"Saved with_angle initial results to {file_path}")

            # Combine with angle results with initial search results
            url_to_info_with_angle_full = { **url_to_info_init_with_angle,**url_to_info}
            file_path = os.path.join(self.args.output_dir, "ref", "with_angle", "with_angle_full", f"ref_{title}.json")
            dump_json(url_to_info_with_angle_full, file_path)
            print(f"Saved with_angle full results to {file_path}")

            print(f"Processed and saved results for article {index + 1}: {title}")
           
        print("All articles processed successfully.")
    



    @log_execution_time
    def _generate_outline_custom(self,
                          json_path:str,#文章的存储位置
                          ):
        """
        Extract the frame from an article.

        Args:
        json_path (str): Path to the JSON file containing article data.
        """
        
        # json_path = "/Users/wangyini/Desktop/code/storm-branch/storm/dataset/articles_new.json"
        article_generate_frame_batch(json_path, self.args.output_dir,0,1)
        #article_generate_frame_one(json_path, output_dir)

            

    @log_execution_time
    def _generate_article_custom(self, 
                                 title = "",
                                 background = "",
                                 base_dir = "/Users/wangyini/Desktop/code/storm-cursor-clean/result-new"
                                ):
    
        
        outline_path = os.path.join(base_dir, "arc", "outline", f"{title}_outline.txt")
        frame_path = os.path.join(base_dir, "arc", "frame", f"{title}_frame.json")
        ref_dir_no_focus = os.path.join(base_dir, "ref", "no_angle","no_angle_full", f"ref_{title}.json")
        ref_dir_with_focus = os.path.join(base_dir, "ref", "with_angle","with_angle_full", f"ref_{title}.json")
        
        
        outline = load_str(outline_path)
        extracted_frame = extract_elements_from_frame(frame_path)

        research_topic = extracted_frame.get("研究主题", "")
        questions = extracted_frame.get("问题", []) # 
        analysis_angles = extracted_frame.get("分析角度", []) #
        

        # Process the outline to get the structure
        outline_tree = process_table_of_contents(outline)
        if len(outline_tree) == 1:
            outline_tree = list(outline_tree.values())[0]

        prefix = ""
        if not self.args.with_focus:
            url_to_info = load_json(ref_dir_no_focus)
            prefix = "no_focus"
        else:
            url_to_info = load_json(ref_dir_with_focus)
            prefix = "with_focus"

        # Process the search results into collected URLs and snippets
        collected_urls = []
        collected_snippets = []
        for url, info in url_to_info.items():
            for snippet in info['webpage_snippets']:
                collected_urls.append(url)
                collected_snippets.append(snippet)

        encoder = SentenceTransformer('BAAI/bge-base-zh-v1.5')
        # Encode snippets
        encoded_snippets = encoder.encode(collected_snippets, show_progress_bar=False)

        # Function to generate each section
        def gen_section(sec_title: str, question: str, focus: List[str], collected_urls: List[str], collected_snippets: List[str]):
            
            """
            Generate a section of the article based on the section title and URL-to-info mapping.
            """
            if not self.args.with_focus:
                # Initialize the section generator
                section_gen = Section_Write_no_focus(engine=self.llm_configs.article_gen_lm)
                generate_retrieve_quesion = DetailedQueryGenerate_no_focus(engine=self.llm_configs.article_gen_lm)

                # Generate detailed queries for the section
                detailed_queries = generate_retrieve_quesion.forward(
                    topic=research_topic,
                    section_theme=sec_title,
                    background=background,
                    question = question
                ).detailed_query

                # Refine search results with SearchCollectedInfo
                search_collected_info = SearchCollectedInfo(
                    collected_urls=collected_urls, 
                    collected_snippets=collected_snippets, 
                    # encoder=encoder,
                    encoded_snippets=encoded_snippets, 
                    search_top_k=self.args.retrieve_top_k,
                    file_path=os.path.join(self.args.output_dir, "article","main_article", prefix, "media_result",sec_title,f"relevant_snippets_{title}.json")
                )

                # Use SearchCollectedInfo to get the most relevant snippets for the section
                url_to_snippets = search_collected_info.search(detailed_queries)

                # Generate the section content
                sec_gen_output = section_gen(
                    question = question,
                    topic=research_topic, 
                    section_theme=sec_title, 
                    background = background,
                    searched_url_to_snippets=url_to_snippets
                )
                sec_result = sec_gen_output.section  # Generated section content
                sec_refs = [{'url': url, 'snippets': url_to_snippets[url]} for url in url_to_snippets]  # References with snippets

            else:

                section_gen = Section_Write_with_focus(engine=self.llm_configs.article_gen_lm)

                # Initialize the SearchQueryGeneratorForChapter with the engine
                search_query_generator = DetailedQueryGenerate_with_focus(engine=self.llm_configs.article_gen_lm)

                # Generate search queries using the forward method
                query_result = search_query_generator.forward(
                    topic=research_topic, 
                    question=question, 
                    section_theme=sec_title, 
                    sub_dimensions=focus, 
                    background=background
                
                )

                # Combine all content into detailed queries
                detailed_queries = [question] + query_result.search_queries

                # Refine search results with SearchCollectedInfo
                search_collected_info = SearchCollectedInfo(
                    collected_urls=collected_urls, 
                    collected_snippets=collected_snippets, 
                    # encoder=encoder,
                    encoded_snippets=encoded_snippets, 
                    search_top_k=self.args.retrieve_top_k,
                    file_path=os.path.join(self.args.output_dir, "article","main_article", prefix, "media_result",sec_title,f"relevant_snippets_{title}.json")

                )

                # Use SearchCollectedInfo to get the most relevant snippets for the section
                url_to_snippets = search_collected_info.search(detailed_queries)

                # Generate the section content using the forward method
                sec_gen_output = section_gen.forward(
                    topic=research_topic, 
                    section_theme=sec_title, 
                    searched_url_to_snippets=url_to_snippets, 
                    question=question, 
                    sub_dimensions=focus, 
                    background=background,
                )

                sec_result = sec_gen_output.section  # Generated section content
                sec_result = clean_incomplete_sentence(sec_result)  # Clean incomplete sentences
                sec_refs = [{'url': url, 'snippets': url_to_snippets[url]} for url in url_to_snippets]  # References with snippets
                reference_urls = list(url_to_snippets.keys())
    
            return sec_result, sec_refs

        # Using concurrent futures to parallelize the generation of sections
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_sec_title = {}
            sections = {}
            search_results = {}

            # Submit tasks for each section
            for section_title, q, a in zip(outline_tree, questions, analysis_angles):
                
                future_to_sec_title[executor.submit(gen_section, section_title, q, a,collected_urls, collected_snippets)] = section_title
                sections[section_title] = None
                search_results[section_title] = None

            # Collect results as they finish
            for future in concurrent.futures.as_completed(future_to_sec_title):
                section_result, section_refs = future.result()
                section_title = future_to_sec_title[future]
                sections[section_title] = section_result
                search_results[section_title] = section_refs

        # Combine all section results into a final article
        sections = list(sections.values())
        search_results = list(search_results.values())
        article = '\n\n'.join(sections)

        # Write output to file
        write_str(article, os.path.join(self.args.output_dir, "article","main_article", prefix, f"{title}.txt"))
        
        return article


    @log_execution_time
    def _polish_article_custom(self,
                           title: str,
                           background: str,
                           base_dir = "/Users/wangyini/Desktop/code/storm-cursor-clean/result-new"
                           ):
        
        """
    Generate the introduction and conclusion of an article and produce a complete polished article.

    Args:
        title (str): Title of the article.
        background (str): Background information for the article.
        base_dir (str): Base directory for article and frame data.

    Returns:
        str: The polished complete article.
    """


        prefix =""
        if self.args.with_focus:
            article_path = os.path.join(base_dir, "article", "main_article", "with_focus", f"{title}.txt")
            prefix = "with_focus"
        else:
            article_path = os.path.join(base_dir, "article", "main_article", "no_focus", f"{title}.txt")
            prefix = "no_focus"

        draft_article = load_str(article_path)
        
        frame_path = os.path.join(base_dir, "arc", "frame", f"{title}_frame.json")
        extracted_frame = extract_elements_from_frame(frame_path)
        themes = extracted_frame.get("主题", []) # 
        
        # Initialize the polishing module
        polish_module = PolishArticleModule(
            intro_engine=self.llm_configs.article_polish_lm,      
            conclusion_engine=self.llm_configs.article_polish_lm, 
        )
        
        gen_result = polish_module(
            topic=themes,
            background=background,
            draft_article=draft_article ,
        )

        
       
        # Clean and process the introduction and conclusion
        introduction = clean_incomplete_sentence(gen_result.introduction)
        conclusion = clean_incomplete_sentence(gen_result.conclusion)

        # Save the complete polished article
        full_path = os.path.join(self.args.output_dir, "article", "full_article", prefix, f"{title}.txt")
        full_draft = f"{introduction}\n\n{draft_article}\n\n{conclusion}"
        write_str(full_draft,full_path )

        return full_draft


    def post_run(self):
        """
        Post-run operations, including:
        1. Dumping the run configuration.
        2. Dumping the LLM call history.
        """
        config_log = self.llm_configs.log()
        dump_json(config_log, os.path.join(self.args.output_dir, "article",self.article_dir_name, 'run_config.json'))

        llm_call_history = self.llm_configs.collect_and_reset_lm_history()
        with open(os.path.join(self.args.output_dir,"article", self.article_dir_name, 'llm_call_history.jsonl'), 'w') as f:
            for call in llm_call_history:
                if 'kwargs' in call:
                    call.pop('kwargs')  # All kwargs are dumped together to run_config.json.
                f.write(json.dumps(call) + '\n')

    def run(self,
            title: str,
            background :str = '',
            do_generate_ref: bool = False,
            do_generate_outline_custom: bool = False,
            do_generate_article_custom: bool = False, 
            do_polish_article_custom: bool = False,
            callback_handler: BaseCallbackHandler = BaseCallbackHandler()):
        
        json_path = "/Users/wangyini/Desktop/code/storm-branch/storm/dataset/articles_new.json" # 原文章存储的路径是固定的
        tavily_api_keys = [
        "XXX"
        ]

        # Generate reference information
        if do_generate_ref:
            self._generate_ref(json_path=json_path,search_top_k = self.args.search_top_k,tavily_api_keys=tavily_api_keys,input_dir=self.args.input_dir,output_dir=self.args.output_dir)
        
        # Generate outline/frame
        if do_generate_outline_custom:
            self._generate_outline_custom(json_path=json_path, output_dir=self.args.output_dir)
            
        # Generate article content
        if do_generate_article_custom:
            self._generate_article_custom(title=title, background=background,base_dir=self.args.base_dir)

        # Polish the article
        if do_polish_article_custom:
            self._polish_article_custom(title=title, background=background)
