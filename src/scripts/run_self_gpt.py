import os 
from argparse import ArgumentParser 
import pandas as pd 
from engine_custom import DeepSearchRunnerArguments, DeepSearchRunner
from modules.utils import MyOpenAIModel, load_api_key, LLMConfigs
from tqdm import tqdm 


def main(args):
    load_api_key() 
    llm_configs = LLMConfigs() 
    llm_configs.init_openai_model(openai_api_key = os.getenv("OPENAI_API_KEY"), openai_type=os.getenv('OPENAI_API_TYPE'),
                                  api_base=os.getenv('API_BASE'))
    
    # Select the model based on the engine type
    if args.engine == 'gpt-4':
         model_name = 'gpt-4'

    # Set the language model for article generation and polishing
    llm_configs.set_article_gen_lm(MyOpenAIModel(model=model_name, api_key=os.getenv("OPENAI_APTI_KEY_4"),
                                                    api_provider=os.getenv("OPENAI_API_TYPE"),
                                                     temperature= float(os.getenv('OPENAI_API_TEMPERATURE', 1.0)), top_p=float(os.getenv("OPENAI_API_TOP_P", 0.9))))

    llm_configs.set_article_polish_lm (MyOpenAIModel(model=model_name, api_key=os.getenv("OPENAI_APTI_KEY_4"),
                                                    api_provider=os.getenv("OPENAI_API_TYPE"),
                                                    temperature=float(os.getenv('OPENAI_API_TEMPERATURE', 1.0)), top_p=float(os.getenv("OPENAI_API_TOP_P", 0.9))))        
    
    # Set up the arguments for the DeepSearchRunner
    engine_args = DeepSearchRunnerArguments(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        search_top_k=args.search_top_k,
        with_focus=args.with_focus,  
        base_dir= args.base_dir

    )

    runner = DeepSearchRunner(engine_args, llm_configs) 


    # Batch processing of articles from the JSON file
    json_path = "articles.json"  # Path to the articles JSON file
    start = 0  # Starting index for reading data
    end = 1  # Ending index (exclusive)

    data = pd.read_json(json_path)  
    data = data.iloc[start:end]  

    
    for _, row in tqdm(data.iterrows(), total=len(data)): 
        title = row['title'] 
        background = row['fact']
        
    
        runner.run(
            title=title,  
            background=background,
            do_generate_ref=args.do_generate_ref,  
            do_generate_outline_custom=args.do_generate_outline_custom,
            do_generate_article_custom=args.do_generate_article_custom,
            do_polish_article_custom=args.do_polish_article_custom,
        )

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--search-top-k', type=int, default=2,
                        help='Top k search results to consider for each search query.')
    
    parser.add_argument('--input-dir', type=str,default='/Users/wangyini/Desktop/code/storm-branch/storm/results/share',
                        help='Using csv file to store topic and ground truth url at present.')
    
    parser.add_argument('--output-dir', type=str, default='/Users/wangyini/Desktop/code/storm-cursor-clean/result-new',
                        help='Directory to store the outputs.')
    parser.add_argument('--engine', type=str, required=True, choices=['gpt-4', 'gpt-3.5-turbo','gpt-4-32k'])
    
    parser.add_argument('--do_generate_ref', action='store_true',
                        help='If True, polish the article by adding a summarization section and (optionally) removing '
                             'duplicate content.')
    parser.add_argument('--do_generate_outline_custom', action='store_true',
                        help='If True, generate article outline/frame from the original article.')
    parser.add_argument('--do_generate_article_custom', action='store_true',
                        help='If True, generate article according frame from the original article.')

    parser.add_argument('--do_polish_article_custom', action='store_true',
                        help='If True, polish the article by adding a summarization section and (optionally) removing '
                             'duplicate content.')

    parser.add_argument('--with_focus', action='store_true',
                        help='If True, use angle-based search queries.')
    parser.add_argument('--base-dir', type=str,default='/Users/wangyini/Desktop/code/storm-cursor-clean/result-new',
                        help='frame reference directory')

    main(parser.parse_args())
