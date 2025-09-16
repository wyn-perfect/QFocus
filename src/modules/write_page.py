"""Write Wikipedia page by simulating conversations to explore the topic."""
import logging
from typing import List, Union, Optional

import dspy
import numpy as np

from modules.utils import limit_word_count_preserve_newline

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import concurrent.futures



class SearchCollectedInfo:
    """Search for the most relevant snippets related to the entries."""
    
    def __init__(self, collected_urls, collected_snippets, encoded_snippets, search_top_k, file_path: str = None):
        self.encoder = SentenceTransformer('BAAI/bge-base-zh-v1.5')  # Initialize the encoder
        self.collected_urls = collected_urls
        self.collected_snippets = collected_snippets  # List of text snippets corresponding to URLs
        self.encoded_snippets = encoded_snippets  # Pre-encoded snippets
        self.search_top_k = search_top_k  # Number of top relevant snippets to select
        self.file_path = file_path
        self.history = []

    def search(self, queries: Union[List[str], str]): 
        instruction = "Generate a representation for this sentence to retrieve relevant articles:"
        selected_urls = []  # URLs to be selected
        selected_snippets = []  # Snippets to be selected
        
        if isinstance(queries, str):  # If query is a string, convert it to a list
            queries = [queries]

        for query in queries:  # Process each query
            encoded_query = self.encoder.encode(instruction + query, show_progress_bar=False)  # Encode the query
            
            # Skip if encoded query or snippets are empty
            if not encoded_query or not self.encoded_snippets:
                print(f"Encoded query or snippets are empty for query: {query}, skipping.")
                continue  # Skip the current query

            sim = cosine_similarity([encoded_query], self.encoded_snippets)[0]  # Compute similarity between query and snippets
            sorted_indices = np.argsort(sim)  # Sort the similarity array

            query_results = {"query": query, "result": {}}  # Store results for the current query
            for i in sorted_indices[-self.search_top_k:][::-1]:  # Select the top k most relevant snippets
                selected_urls.append(self.collected_urls[i])
                selected_snippets.append(self.collected_snippets[i])

                url = self.collected_urls[i]
                snippet = self.collected_snippets[i]
                if url not in query_results["result"]:
                    query_results["result"][url] = []
                query_results["result"][url].append(snippet)

            # Remove duplicate snippets for each URL
            for url in query_results["result"]:
                query_results["result"][url] = list(set(query_results["result"][url]))

            # Add the current query results to history
            self.history.append(query_results)
        
        # Store the query results in a dictionary
        url_to_snippets = {}
        for url, snippet in zip(selected_urls, selected_snippets):
            if url not in url_to_snippets:
                url_to_snippets[url] = []
            url_to_snippets[url].append(snippet)

        # Remove duplicates and return the final results
        for url in url_to_snippets:
            url_to_snippets[url] = list(set(url_to_snippets[url]))

        return url_to_snippets  

class WriteSection_no_focus(dspy.Signature):
    """
    你是一个优秀的文章撰写者，能够根据提供的研究主题、背景信息、关键主题和收集到的信息，撰写深度的分析章节
    撰写章节的目的是通过围绕给定的关键主题进行深入分析从而回答给定的主要问题。
    章节内容应结合整篇文章的研究主题和背景信息，但重点应放在围绕关键主题完成章节的撰写目的。分析应有理有据，包含相关的事实、数据和专家解读等作为支撑。
    章节内容清晰，逻辑严谨，包含分析的结论和论据。
    章节应分为1-2段，每段字数在150至500字之间
    
    章节应包含以下要点：
    1. 生成一个精炼的小标题，概括章节的核心观点或结论，吸引读者的注意力。
    2. 根据提供的该章节的撰写目的，围绕提供的关键主题，结合所有的相关信息，进行深入分析，确保分析有力有据
    3. 包含相关事实和数据，用事实、数据和专家解读支撑分析的结论，确保内容真实可靠。
    4. 保持内容严谨，避免冗余，语言专业犀利。
    5. 逻辑清晰地呈现事实和数据，支撑分析的结论，不能虚构内容
    6. 正文部分不要添加任何其他与分析无关的解释说明,不要有其他额外的解释或者评价

    输出格式：
    -标题：生成一个小标题（仅一个），直接用 "##" 开头，概括章节的核心观点或结论。
    -正文：生成正文内容部分，确保分为自然段，不使用其他章节标题（如 "##"、"###" 等）。正文内容应在150至500字之间。
    
    """

    # Input fields
    
    topic = dspy.InputField(prefix="整篇文章的研究主题: ", format=str)  # The research question
    background = dspy.InputField(prefix="整篇文章的主题的背景信息", format=str) 
    info = dspy.InputField(prefix="收集到的信息:\n", format=str)  # The information collected from the sources
    question = dspy.InputField(prefix="该章节撰写的目的/需要回答的问题 ", format=str) 
    section_theme = dspy.InputField(prefix="该章节的关键主题（议题）: ", format=str)  # The perspective for analysis
    output = dspy.OutputField(
        prefix="根据提供的信息、研究主题、关键主题和主要问题，撰写分析章节:\n"
    )
    

class Section_Write_no_focus(dspy.Module):
    """
    Generate an analysis section and a catchy subheading based on the given topic, 
    section theme, question, background, and collected snippet information.
    """

    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.write_section = dspy.Predict(WriteSection_no_focus)  # Predictor for writing section content
        self.engine = engine  # Language model engine

    def forward(self, topic: str, background: str, section_theme: str, question: str, searched_url_to_snippets: dict):
        # Combine all snippets into a single info string
        info = ''
        for n, r in enumerate(searched_url_to_snippets.values()):
            info += f'[{n + 1}]\n' + '\n'.join(r) + '\n\n'

        # Limit total word count for prompt context
        info = limit_word_count_preserve_newline(info, 4000)

        # Generate section content using the specified LLM engine
        with dspy.settings.context(lm=self.engine):
            output = self.write_section(
                topic=topic,
                info=info,
                section_theme=section_theme,
                question=question,
                background=background
            )
            section = output.output  # Extract generated section content

        return dspy.Prediction(section=section)





class WriteSection_with_focus(dspy.Signature):
    """
    你是一个优秀的文章撰写者，擅长根据提供的研究主题、背景信息、相关信息、分析角度及关键子维度，撰写精准且深入的分析章节。
    你的任务是根据该章节的撰写目的或该章节要回答的问题，紧紧围绕分析视角以及关键子维度，通过严谨的逻辑和详实的论据，提供有深度、有说服力的分析内容。
    请确保分析有理有据，结合事实、论据、相关信息和专家解读进行支撑，避免虚构或重复内容。
    章节应分为1-2段，每段字数在150-500字左右
    
    章节应包含以下要点：
    1. 生成一个精炼的小标题，能够精准概括章节的核心观点或结论，吸引读者的注意力。
    2. 紧扣撰写目的与分析角度：结合研究主题、背景信息、收集的信息（提供的事实论据），进行深入的解读，确保分析有理有据。
    3. 深入分析关键子维度：对每个子维度进行充分详细讨论，避免表面化描述，深入挖掘问题本质。
    4. 使用提供的事实、论据和相关信息、专家解读等来支撑分析结论，确保细节丰富、逻辑清晰。
    5. 保持内容严谨，避免冗余描述，不得虚构内容，不得重复生成。
    6. 正文内容中逻辑清晰地呈现观点，提供强的事实和数据，支撑分析的结论，不得虚构。
    7. 正文部分不要添加任何其他与分析无关的解释说明,不要有其他额外的解释或者评价

    输出格式：
    -标题：生成一个小标题（仅一个），直接用 "##" 开头，概括章节的核心观点或结论。
    -正文：生成正文内容部分，确保分为自然段，不得使用其他章节标题（如 "##"、"###" 等）。正文内容应在150至500字之间。
    
    """

    
    topic = dspy.InputField(prefix="研究主题：", format=str)  
    background = dspy.InputField(prefix="背景信息：", format=str)  
    info = dspy.InputField(prefix="相关信息（事实论据）：\n", format=str) 
    section_theme = dspy.InputField(prefix="关键主题（分析的议题）：", format=str)  
    sub_dimensions = dspy.InputField(prefix="关键子维度：", format=list)  
    question = dspy.InputField(prefix="该章节撰写的目的或者是回答的问题", format=str) 
    output = dspy.OutputField(
        prefix="根据提供的信息、研究主题和关键主题以及关键子维度 和撰写本章节的目的，撰写分析章节内容：\n"
    )  


class Section_Write_with_focus(dspy.Module):
    """
    Generate an analysis section and a catchy subheading based on the collected information, 
    analysis perspective, and research question.
    """
    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.write_section = dspy.Predict(WriteSection_with_focus)  # Predictor for generating section content
        self.engine = engine  # Language model engine

    def forward(self, topic: str, section_theme: str, searched_url_to_snippets: dict, 
                question: str, sub_dimensions: list, background: str):
        """
        Generate the section content and subheading based on the provided topic, 
        analysis perspective, collected snippets, and relevant background information.
        
        Args:
            topic: The research topic.
            section_theme: The analysis perspective (focus of the section).
            searched_url_to_snippets: Collected URLs and snippets from the search results.
            question: The specific question the section aims to answer.
            sub_dimensions: The sub-dimensions or key aspects of the analysis.
            background: The background information for the section.
        
        Returns:
            The generated section content and subheading.
        """
        
        # Combine the collected snippets into a single info string
        info = ''
        for n, r in enumerate(searched_url_to_snippets.values()):
            info += f'[{n + 1}]\n' + '\n'.join(r)  # Merge snippets for each URL
            info += '\n\n'

        # Limit the word count to ensure context is preserved
        info = limit_word_count_preserve_newline(info, 4000)  # Restrict word count to maintain context integrity

        # Generate section content and subheading using the language model
        with dspy.settings.context(lm=self.engine):
            output = self.write_section(
                topic=topic,  # The research topic
                section_theme=section_theme,  # The key theme (focus of the analysis)
                info=info,  # The collected information
                question=question,  # The specific question for the section
                sub_dimensions=sub_dimensions,  # The sub-dimensions of the section
                background=background  # Background information
            )
            section = output.output  # Extract generated section content
        
        return dspy.Prediction(section=section)  # Return the generated section




####################################
###### polish artical custom
####################################

class WriteIntroduction(dspy.Signature):
    """你的任务是根据提供的据研究主题、正文内容和背景信息，为该文章撰写简洁的引言。
    引言应满足以下要求：
    1. 明确陈述研究主题，并结合背景数据（如事件时间、影响规模）强调研究的重要性。
    2. 简要概述正文部分涉及的主要内容和分析视角，并提示即将得出的结论。
    3. 语言简洁有力，避免冗余内容，字数控制在150-200字，以一段自然段的形式撰写。
    4. 不要有任何的标题前缀，仅输出引言部分，不添加其他内容或格式，也不要生成任何其他的解释性说明或评价。
    """
    topic = dspy.InputField(prefix="研究主题:", format=str)
    background = dspy.InputField(prefix="背景信息:", format=str)
    draft_article = dspy.InputField(prefix="正文部分）:\n", format=str)
    introduction = dspy.OutputField(prefix="生成引言:\n", format=str)


class WriteConclusion(dspy.Signature):
    """
    你的任务是根据提供的研究主题、背景信息以及正文部分，撰写文章的结尾部分。结尾部分应包括：
    1. 总结核心发现。
    2. 提供明确的政策建议或未来研究方向，确保建议基于正文中的分析依据。
    3. 结论部分应简洁明了，以自然段的方式撰写，控制在1-2段内。
    4. 不要有任何的标题前缀，仅输出结论部分，不添加其他内容或格式，也不要生成任何其他的解释性说明。
    """
    topic = dspy.InputField(prefix="研究主题:", format=str)
    background = dspy.InputField(prefix="背景信息:", format=str)
    draft_article = dspy.InputField(prefix="正文部分：\n", format=str)
    conclusion = dspy.OutputField(prefix="生成结论：\n", format=str)


class PolishArticleModule(dspy.Module):
    def __init__(self, intro_engine: Union[dspy.dsp.LM, dspy.dsp.HFModel],
                 conclusion_engine: Union[dspy.dsp.LM, dspy.dsp.HFModel],
                 ):
        super().__init__()
        self.intro_engine = intro_engine
        self.conclusion_engine = conclusion_engine
        self.write_intro = dspy.Predict(WriteIntroduction)
        self.write_conclusion = dspy.Predict(WriteConclusion)

    def forward(self, topic: str, background: str, draft_article: str): 
        with dspy.settings.context(lm=self.intro_engine):
            intro = self.write_intro(
                topic=topic, 
                background=background, 
                draft_article=draft_article
            ).introduction
            intro = self._clean_output(intro, "生成引言:")  
            

        with dspy.settings.context(lm=self.conclusion_engine):
            conclusion = self.write_conclusion(
                topic=topic,
                background=background,
                draft_article=draft_article,
            ).conclusion
            conclusion = self._clean_output(conclusion, "生成结论:")

    
        return dspy.Prediction(
            introduction=intro, 
            conclusion=conclusion, 
               
        )
    
    def _clean_output(self, text: str, prefix: str) -> str:
        return text.split(prefix)[-1].strip() if prefix in text else text

