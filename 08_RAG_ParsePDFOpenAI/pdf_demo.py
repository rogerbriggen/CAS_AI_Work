''' 
This script demonstrates how to process PDFs with OpenAI.
It extracts text and images from PDF files and sends the images to OpenAI for analysis.
The results are saved as a JSON file.
'''
from ast import literal_eval
import concurrent
import json
import re
import sys
import os
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from pdf_preprocess import PDFPreprocess
from openai_wrapper import OpenAIWrapper


class PDFDemo:
    ''' 
    This class demonstrates how to process PDFs with OpenAI.
    It extracts text and images from PDF files and sends the images to OpenAI for analysis.
    The results are saved as a JSON file.
    '''

    def extract_slides_text_from_pdf(self, pdf_files_path, max_files = 2):
        '''
        Extract the text and from the pdf and saves that as text property.
        It also extracts the images from the pdf and sends it to openai for analysis with the instructions to create slides from the images. 
        The results are saved as pages_description property.
        pdf_files_path: str, path to the PDF files
        max_files: int, maximum number of files to process
        '''
        docs = []
        pdf_preprocess = PDFPreprocess()
        openai_wrapper = OpenAIWrapper()
        all_items = os.listdir(pdf_files_path)
        files = [item for item in all_items if os.path.isfile(os.path.join(pdf_files_path, item))]
        for f in files[0:max_files]:

            path = f"{pdf_files_path}/{f}"
            doc = {
                "filename": f
            }
            text = pdf_preprocess.extract_text_from_doc(path)
            doc['text'] = text
            imgs = pdf_preprocess.convert_to_images(path)
            pages_description = []

            print(f"Analyzing pages for doc {f}")

            # Concurrent execution
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:

                # Removing 1st slide as it's usually just an intro
                futures = [
                    executor.submit(openai_wrapper.analyze_doc_image, img)
                    for img in imgs[1:]
                ]

                with tqdm(total=len(imgs)-1) as pbar:
                    for _ in concurrent.futures.as_completed(futures):
                        pbar.update(1)

                for f in futures:
                    res = f.result()
                    pages_description.append(res)

            doc['pages_description'] = pages_description
            docs.append(doc)
        print("Finished processing all docs")
        return docs


    def save_docs_as_json(self, docs, output_path = "temp/pdf_demo/parsed_pdf_docs.json"):
        '''
        Save a list of documents to a JSON file.
        docs: list, list of documents
        output_path: str, path to save the JSON file
        '''
        # create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # save to json
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(docs, f)


    def load_docs_from_json(self, input_path = "temp/pdf_demo/parsed_pdf_docs.json"):
        '''
        Load a list of documents from a JSON file.
        input_path: str, path to the JSON file to load the documents from
        '''
        with open(input_path, "r", encoding="utf-8") as f:
            docs = json.load(f)
        return docs

    def embedding_content(self, docs):
        '''
        Extracts the content from the docs and returns a list of strings.
        docs: list, list of documents
        '''
        content = []
        for doc in docs:
            # Removing first slide as well
            text = doc['text'].split('\f')[1:]
            description = doc['pages_description']
            description_indexes = []
            for i in range(len(text)): # pylint: disable=consider-using-enumerate
                slide_content = text[i] + '\n'
                # Trying to find matching slide description
                slide_title = text[i].split('\n')[0]
                for j in range(len(description)): # pylint: disable=consider-using-enumerate
                    description_title = description[j].split('\n')[0]
                    if slide_title.lower() == description_title.lower():
                        slide_content += description[j].replace(description_title, '')
                        # Keeping track of the descriptions added
                        description_indexes.append(j)
                # Adding the slide content + matching slide description to the content pieces
                content.append(slide_content)
            # Adding the slides descriptions that weren't used
            for j in range(len(description)): # pylint: disable=consider-using-enumerate
                if j not in description_indexes:
                    content.append(description[j])
        return content


    def cleanup_content(self, content):
        '''
        Cleans up the content by removing trailing spaces, additional line breaks, page numbers and references to the content being a slide
        content: list, list of strings
        '''
        clean_content = []
        for c in content:
            text = c.replace(' \n', '\n').replace('\n\n', '\n').replace('\n\n\n', '\n').strip()
            text = re.sub(r"(?<=\n)\d{1,2}", "", text)
            text = re.sub(r"\b(?:the|this)\s*slide\s*\w+\b", "", text, flags=re.IGNORECASE)
            clean_content.append(text)
        return clean_content
    
    
    def create_embeddings(self, docs, embeddings_file_path):
        '''
        Extracts the content from the docs, cleans it up and returns a list of strings.
        docs: list, list of documents
        '''
        content = self.embedding_content(docs)
        clean_content = self.cleanup_content(content)
        # Creating the embeddings
        df = pd.DataFrame(clean_content, columns=['content'])
        print(df.shape)
        print(f'first rows of the cleaned content: {df.head()}')
        openai_wrapper = OpenAIWrapper()
        df['embeddings'] = df['content'].apply(lambda x: openai_wrapper.get_embeddings(x))
        print(f'first rows of the cleaned context and embeddings: {df.head()}')
        # Saving locally for later... should go to a vector database
        df.to_csv(embeddings_file_path, index=False)

    def load_embeddings(self, embeddings_file_path):
        '''
        Load embeddings from a csv file.
        '''
        df = pd.read_csv(embeddings_file_path)
        df["embeddings"] = df.embeddings.apply(literal_eval).apply(np.array)
        return df
    

    def rag_search_content(self, df_content, input_text, top_k):
        '''	
        Search for the most similar content to the input text in the dataframe.
        df_content: pd.DataFrame, dataframe with the content and embeddings
        input_text: str, input text to search for
        top_k: int, number of most similar content to return
        '''
        openai_wrapper = OpenAIWrapper()
        embedded_value_input_text = openai_wrapper.get_embeddings(input_text)
        df_content["similarity"] = df_content.embeddings.apply(lambda x: cosine_similarity(np.array(x).reshape(1,-1), np.array(embedded_value_input_text).reshape(1, -1)))
        res = df_content.sort_values('similarity', ascending=False).head(top_k)
        return res

    def rag_get_similarity(self, row):
        '''
        Get the similarity score from a row.
        row: pd.Series, row from the dataframe
        '''
        if isinstance(row, tuple):
            similarity_score = row[1]['similarity']
        else:
            similarity_score = row['similarity']
        if isinstance(similarity_score, np.ndarray):
            similarity_score = similarity_score[0][0]
        return similarity_score

    def rag_generate_output(self, input_prompt, similar_content, threshold = 0.5):
        '''
        Generate an output based on the input prompt and the most similar content.
        input_prompt: str, input prompt
        similar_content: pd.DataFrame, dataframe with the most similar content
        threshold: float, similarity threshold to add more content
        '''

        content = similar_content.iloc[0]['content']

        # Adding more matching content if the similarity is above threshold
        if len(similar_content) > 1:
            for row in similar_content.iterrows():
                if isinstance(row, tuple):
                    row = row[1]
                similarity_score = self.rag_get_similarity(row)
                if similarity_score > threshold:
                    content += f"\n\n{row['content']}"

        prompt = f"INPUT PROMPT:\n{input_prompt}\n-------\nCONTENT:\n{content}"
        openai_wrapper = OpenAIWrapper()
        completion = openai_wrapper.openai.chat_completions_create(
            model="gpt-4o",
            temperature=0.5,
            messages=[
                {
                    "role": "system",
                    "content": openai_wrapper.system_prompt_text
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        print(f"Completion choices: {completion.choices}")
        return completion.choices[0].message.content

    def run_rag(self, embeddings_file):
        '''
        Run the RAG pipeline on example user queries related to the content.
        embeddings_file: str, path to the csv file with the embeddings
        '''

        # Example user queries related to the content
        example_inputs = [
            'What are the main models you offer?',
            'Do you have a speech recognition model?',
            'Which embedding model should I use for non-English use cases?',
            'Can I introduce new knowledge in my LLM app using RAG?',
            'How many examples do I need to fine-tune a model?',
            'Which metric can I use to evaluate a summarization task?',
            'Give me a detailed example for an evaluation process where we are looking for a clear answer to compare to a ground truth.',
        ]
        df = self.load_embeddings(embeddings_file)
        # Running the RAG pipeline on each example
        for ex in example_inputs:
            print(f"QUERY: {ex}\n\n")
            matching_content = self.rag_search_content(df, ex, 3)
            print("Matching content:\n")
            for match in matching_content.iterrows():
                if isinstance(match, tuple):
                    match = match[1]
                print(f"Similarity: {self.rag_get_similarity(match):.2f}")
                print(f"{match['content'][:100]}{'...' if len(match['content']) > 100 else ''}\n\n")
            reply = self.rag_generate_output(ex, matching_content)
            print(f"REPLY:\n\n{reply}\n\n--------------\n\n")


def main():
    str_process_all = "processAll"
    str_create_embeddings = "createEmbeddings"
    str_run_rag = "runRag"
    str_extract_text = "extractText"

    # Initialize parser
    parser = argparse.ArgumentParser(description="Demo how to process PDFs with opanai")

    # Adding optional argument
    parser.add_argument("-pt", "--processType", required=False, default="processAll", choices=[str_extract_text, str_create_embeddings, str_run_rag, str_process_all], help = "createEmbeddings: loads the docs which were already preprocessed and creates embeddings")
    parser.add_argument("-pdf", "--pdfPath", required=False, help = "Path to the PDF files to process")
    parser.add_argument("-j", "--jsonFile", required=False, help = "Path and name of the JSON file used to save/load the docs")
    parser.add_argument("-e", "--embeddingsFile", required=False, help = "Path and name of the csv file used to save/load the embeddings")


    # Read arguments from command line
    args = parser.parse_args()

    print(f'pdfPath: {args.pdfPath}')
    print(f'processType: {args.processType}')
    print(f'jsonFile: {args.jsonFile}')
    print(f'embeddingsFile: {args.embeddingsFile}')


    current_dir = os.path.dirname(os.path.abspath(__file__))
    if args.pdfPath:
        pdf_files_path = args.pdfPath
    else :
        pdf_files_path = os.path.join(current_dir, "data/example_pdfs/")

    if args.jsonFile:
        json_file = args.jsonFile
    else:
        json_file = os.path.join(current_dir, "temp/pdf_demo/parsed_pdf_docs.json")

    if args.embeddingsFile:
        embeddings_file = args.embeddingsFile
    else:
        embeddings_file = os.path.join(current_dir, "temp/pdf_demo/parsed_pdf_docs_with_embeddings.csv")

    pdf_demo = PDFDemo()

    if args.processType == str_extract_text:
        docs = pdf_demo.extract_slides_text_from_pdf(pdf_files_path=pdf_files_path)
        pdf_demo.save_docs_as_json(docs, json_file)
        print(f"Saved {len(docs)} docs")
    elif args.processType == str_create_embeddings:
        docs = pdf_demo.load_docs_from_json(json_file)
        print(f"Loaded {len(docs)} docs")
        pdf_demo.create_embeddings(docs, embeddings_file)
        print(f"Saved embeddings to {embeddings_file}")
    elif args.processType == str_run_rag:
        pdf_demo.run_rag(embeddings_file)
        print("RAG finished")
    elif args.processType == str_process_all:
        # extract text
        docs = pdf_demo.extract_slides_text_from_pdf(pdf_files_path=pdf_files_path, max_files=4)
        pdf_demo.save_docs_as_json(docs, json_file)
        print(f"Saved {len(docs)} docs")
        # create embeddings
        docs = pdf_demo.load_docs_from_json(json_file)
        print(f"Loaded {len(docs)} docs")
        pdf_demo.create_embeddings(docs, embeddings_file)
        print(f"Saved embeddings to {embeddings_file}")
        # run rag
        pdf_demo.run_rag(embeddings_file)
        print("RAG finished")
    else:
        print("Invalid process type")
        sys.exit(1)

if __name__ == "__main__":
    main()
