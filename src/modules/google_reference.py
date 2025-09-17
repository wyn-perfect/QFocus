import os
import re
from typing import Union
from modules.rm import TavilySearchRM  
import dspy

script_dir = os.path.dirname(os.path.abspath(__file__))




# no question and focus

# generate search queries for article sections
class GenerateSearchQueries_no_focus(dspy.Signature):
    """
    您的任务是为撰写文章的特定章节生成有效的搜索查询。
    这些查询将围绕研究主题、相关事实、以及关键主题展开，以收集高质量的参考资料。
    请确保生成的查询：
    1. **具体且针对性强**：聚焦于研究主题的关键主题，避免过于宽泛的关键词堆砌。
    2. **以问题形式呈现**：尽量以“如何……”、“为什么……”、“……的影响是什么？”等形式生成查询，以便直接获取相关答案。
    3. **结合背景信息和关键主题**：反映研究主题的背景和关键主题，确保查询的针对性。
    4. **结合所有输入内容**：生成的查询要充分融入研究主题、背景信息和章节核心主题，确保搜索结果最大程度地贴合目标内容。
    
    并按照以下格式编写：
        查询1:
        查询2:
        ……
        查询n: 
    """
    
    topic = dspy.InputField(prefix="研究主题: ", format=str)
    background = dspy.InputField(prefix="背景(研究主题的相关事实): ", format=str)
    section_theme = dspy.InputField(prefix="该章节的关键主题: ", format=str)
    queries = dspy.OutputField(prefix="生成的搜索条目:", format=str)


class SearchWithExclusions_no_focus(dspy.Module):
    """1. Generate search queries based on topic, background, section_theme, and exclude specific URLs. 
       2. Perform search using the generated queries and exclude certain URLs."""
    

    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel], search_top_k: int,api_key: str,):
        super().__init__()
        self.generate_queries = dspy.Predict(GenerateSearchQueries_no_focus)
        self.retrieve = TavilySearchRM(k=search_top_k,tavily_search_api_key=api_key)  # Assuming TavilySearchRM is your retrieval class
        self.engine = engine

    def forward(self, topic: str, background: str, section_theme: str, ground_truth_url: str):
        with dspy.settings.context(lm=self.engine):
            # Generate search queries based on the input fields
            queries = self.generate_queries(topic=topic, background=background, section_theme=section_theme).queries
            queries = re.findall(r'\d+\:\s*(.*?)\n', queries)
            queries = list(set(queries))
            
            #queries = [q.replace('-', '').strip().strip('"').strip('"').strip() for q in queries.split('\n')][:2]
            # Perform search with the generated queries and exclude the given URLs
            searched_results = self.retrieve(list(set(queries)), exclude_urls=[ground_truth_url])

        # Return the generated queries and the retrieved search results
        return dspy.Prediction(queries=queries, searched_results=searched_results)




# generate detailed queries for article sections
class GenerateDetailedQuery_no_focus(dspy.Signature):
    """
       您是一位专业的研究助理，能够基于一个主要问题，结合研究主题、背景信息和特定子主题，生成多个相关的额外问题。
       这些问题将帮助您收集高质量的信息，以支持章节的分析。由于资料大多是新闻类或报道类，相关的事实、数据和专家解读非常重要。
       生成的额外问题应紧密围绕特定子主题，这些子主题是回答主问题的关键切入点。每个额外问题都应有助于深入理解和回答主问题。
       请按照以下格式编写问题：
            问题1:
            问题2:
            ……
            问题n:
    """
    topic = dspy.InputField(prefix="研究主题: ", format=str)
    section_theme = dspy.InputField(prefix="该章节的关键主题: ", format=str)
    background = dspy.InputField(prefix="主题的背景信息）: ", format=str)
    question = dspy.InputField(refix="主要问题",format=str)
    detailed_query = dspy.OutputField(prefix="生成的额外问题： ", format=str)


class DetailedQueryGenerate_no_focus(dspy.Module):
    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.generate_queries = dspy.Predict(GenerateDetailedQuery_no_focus)
        self.engine = engine

    def forward(self, topic: str, section_theme: str, background: str, question: str):
        """
        Generates a more detailed query by combining the topic and the research angle (sub-topic).
        Args:
            topic: The main research topic.
            section_theme: sub-topic of the research.
            background: The background context for the research.
        Returns:
            A detailed query string to use for search.
        """
        with dspy.settings.context(lm=self.engine):

            query_result = self.generate_queries(topic=topic, section_theme=section_theme, background=background, question=question).detailed_query
            query_result = re.findall(r'\d+\:\s*(.*?)\n', query_result)
            return dspy.Prediction(detailed_query=query_result)



# with focus

class GenerateSearchQueries_with_focus(dspy.Signature):
    """
    您的任务是为撰写文章的特定章节生成有效的搜索查询，以收集高质量的参考资料。
    这些查询将围绕特定问题，从特定角度出发，并聚焦于关键子维度。
    
    请确保生成的查询：
    1. **具体且针对性强**：聚焦于核心问题和子维度，避免过于宽泛的关键词堆砌。
    2. **以问题形式呈现**：尽量以“如何……”、“为什么……”、“……的影响是什么？”等形式生成查询。
    3. **结合背景信息和分析角度**：反映研究主题的背景和关键主题，确保查询的针对性。
    4. **结合所有输入内容**：确保查询中包含研究主题、问题、关键主题、子维度和背景信息，以生成最相关的搜索结果。

    请按照以下格式编写将要使用的查询：
        1. 查询1
        2. 查询2
        ...
        n. 查询n
    """

    topic = dspy.InputField(prefix='文章的研究主题：', format=str)
    question = dspy.InputField(prefix='该章节所回答的具体问题：', format=str)
    section_theme = dspy.InputField(prefix='该章节的关键主题：', format=str)
    sub_dimensions = dspy.InputField(
        prefix='该章节所关注的子维度（以列表形式提供）：', format=list)
    background_info = dspy.InputField(prefix='关于主题的背景信息：', format=str)
    search_entries = dspy.OutputField(
        prefix='生成一系列搜索查询，帮助精准收集与该章节相关的高质量资料。'
               '查询应具体且有针对性，避免使用过于宽泛或不相关的术语。\n')
    


class SearchWithExclusions_with_focus(dspy.Module):
    """This module generates search queries based on topic, background, perspective,sub_dimensions， and excludes specific URLs. 
    It performs a search using the generated queries and excludes certain URLs from the results."""

    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel], search_top_k: int,api_key: str,):
        super().__init__()
        self.generate_queries = dspy.Predict(GenerateSearchQueries_with_focus)
        self.retrieve = TavilySearchRM(k=search_top_k,tavily_search_api_key=api_key)  # Assuming TavilySearchRM is your retrieval class

        self.engine = engine

    def forward(self, topic: str, question: str, section_theme: str, sub_dimensions: list, background_info: str, ground_truth_url: str):
        with dspy.settings.context(lm=self.engine):
            # Generate search queries using the GenerateSearchEntries module
            queries = self.generate_queries(
                topic=topic,
                question=question,
                section_theme=section_theme,
                sub_dimensions=sub_dimensions,
                background_info=background_info
            ).search_entries

            # Display the original generated queries
            
            
            queries = re.findall(r'\d+\.\s*(.*?)\n', queries)
            # Extract and clean the queries using regular expressions
            queries = list(set(queries))
            # Perform the search with the generated queries and exclude the given URLs
            searched_results = self.retrieve(
                list(set(queries)), exclude_urls=[ground_truth_url]
            )

        # Return the generated queries and the retrieved search results
        return dspy.Prediction(queries=queries, searched_results=searched_results)




class GenerateDetailedQuery_with_focus(dspy.Signature):
    """
    您是一位专业的研究助理，能够基于一个主要问题，结合研究主题、背景信息、特定主题和具体的子角度，生成多个相关的额外问题。
    这些问题将帮助您收集高质量的信息，以支持章节的分析。由于资料大多是新闻类或报道类，相关的事实、数据和专家解读非常重要。
    生成的额外问题应紧密围绕特定主题和子角度，这些主题和子角度是回答主问题的关键切入点。每个额外问题都应有助于深入理解和回答主问题。

    请按照以下格式编写额外问题：
        问题1:
        问题2:
        ……
        问题n:
    """

    topic = dspy.InputField(prefix="研究主题:", format=str)
    question = dspy.InputField(prefix="该章节需要回答的主要问题:", format=str)
    section_theme = dspy.InputField(prefix="该章节的关键主题:", format=str)
    sub_dimensions = dspy.InputField(
        prefix="该章节所关注的具体子角度或关键方面（以逗号分隔）:", format=list)
    background_info = dspy.InputField(prefix="背景信息:", format=str)
    search_queries = dspy.OutputField(
        prefix="生成的额外问题：", format=str)
    
class DetailedQueryGenerate_with_focus(dspy.Module):
    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.generate_queries = dspy.Predict(GenerateDetailedQuery_with_focus)
        self.engine = engine

    def forward(self, topic: str, question: str, section_theme: str, sub_dimensions: list, background: str):
        """
        Generates focused search queries that help identify the most relevant content for the chapter.
        
        Args:
            topic: The main research topic of the chapter.
            question: The specific question that the chapter is answering.
            section_theme: topic of the chapter.
            sub_dimensions: A list of sub-dimensions or key aspects the chapter focuses on.
            background: The background context for the chapter.
            
        Returns:
            A list of focused search queries to be used for retrieving relevant content for the chapter.
        """
        with dspy.settings.context(lm=self.engine):
    
            # Generate search queries based on the provided inputs
            query_result = self.generate_queries(
                topic=topic,
                question=question,
                section_theme=section_theme,
                sub_dimensions=sub_dimensions,
                background_info=background
            ).search_queries
            
            query_result = re.findall(r'\d+[.:]\s*(.*?)\n', query_result)

            return dspy.Prediction(search_queries=query_result)












