''' 
This script demonstrates how to process PDFs with OpenAI.
It extracts text and images from PDF files and sends the images to OpenAI for analysis.
The results are saved as a JSON file.
'''
import concurrent
import json
import sys
import os
import argparse

from tqdm import tqdm
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



def main():
    str_process_all = "processAll"
    str_load_only = "loadOnly"

    # Initialize parser
    parser = argparse.ArgumentParser(description="Demo how to process PDFs with opanai")

    # Adding optional argument
    parser.add_argument("-pt", "--processType", required=False, default="processAll", choices=[str_process_all, str_load_only], help = "loadOnly: loads the docs which were already preprocessed")
    parser.add_argument("-p", "--pdfPath", required=False, help = "Path to the PDF files to process")
    parser.add_argument("-j", "--jsonFile", required=False, help = "Path and name of the JSON file used to save/load the docs")


    # Read arguments from command line
    args = parser.parse_args()

    print(f'pdfPath: {args.pdfPath}')
    print(f'processType: {args.processType}')
    print(f'jsonFile: {args.jsonFile}')


    current_dir = os.path.dirname(os.path.abspath(__file__))
    if args.pdfPath:
        pdf_files_path = args.pdfPath
    else :
        pdf_files_path = os.path.join(current_dir, "data/example_pdfs/")

    if args.jsonFile:
        json_file = args.jsonFile
    else:
        json_file = os.path.join(current_dir, "temp/pdf_demo/parsed_pdf_docs.json")

    pdf_demo = PDFDemo()

    if args.processType == str_load_only:
        docs = pdf_demo.load_docs_from_json(json_file)
        print(f"Loaded {len(docs)} docs")
    elif args.processType == str_process_all:
        docs = pdf_demo.extract_slides_text_from_pdf(pdf_files_path=pdf_files_path)
        pdf_demo.save_docs_as_json(docs, json_file)
        print(f"Saved {len(docs)} docs")
    else:
        print("Invalid process type")
        sys.exit(1)

if __name__ == "__main__":
    main()
