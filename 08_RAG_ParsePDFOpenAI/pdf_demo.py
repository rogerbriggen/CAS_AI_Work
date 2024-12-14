# process the demo pdfs
import concurrent
import json

from tqdm import tqdm
from pdf_preprocess import PDFPreprocess
from openai_wrapper import OpenAIWrapper
import sys
import os
import argparse

class PDFDemo:
       
        
    '''
    Convert a PDF file to a list of images.
    pdf_files_path: str, path to the PDF files
    max_files: int, maximum number of files to process
    '''
    def process_pdfs(self, pdf_files_path, max_files = 2):
        docs = []
        pdfPreprocess = PDFPreprocess()
        openAIWrapper = OpenAIWrapper()
        all_items = os.listdir(pdf_files_path)
        files = [item for item in all_items if os.path.isfile(os.path.join(pdf_files_path, item))]
        for f in files[0:max_files]:

            path = f"{pdf_files_path}/{f}"
            doc = {
                "filename": f
            }
            text = pdfPreprocess.extract_text_from_doc(path)
            doc['text'] = text
            imgs = pdfPreprocess.convert_to_images(path)
            pages_description = []

            print(f"Analyzing pages for doc {f}")

            # Concurrent execution
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:

                # Removing 1st slide as it's usually just an intro
                futures = [
                    executor.submit(openAIWrapper.analyze_doc_image, img)
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

    '''
    Save a list of documents to a JSON file.
    docs: list, list of documents
    output_path: str, path to save the JSON file
    '''
    def save_docs_as_json(self, docs, output_path = "temp/pdf_demo/parsed_pdf_docs.json"):
        # create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # save to json
        with open(output_path, "w") as f:
            json.dump(docs, f)

    '''
    Load a list of documents from a JSON file.
    input_path: str, path to the JSON file to load the documents from
    '''
    def load_docs_from_json(self, input_path = "temp/pdf_demo/parsed_pdf_docs.json"):
        with open(input_path, "r") as f:
            docs = json.load(f)
        return docs

    

def main():
    str_processAll = "processAll"
    str_loadOnly = "loadOnly"

    # Initialize parser
    parser = argparse.ArgumentParser(description="Demo how to process PDFs with opanai")

    # Adding optional argument
    parser.add_argument("-pt", "--processType", required=False, default="processAll", choices=[str_processAll, str_loadOnly], help = "loadOnly: loads the docs which were already preprocessed")
    parser.add_argument("-p", "--pdfPath", required=False, help = "Path to the PDF files to process")
    parser.add_argument("-j", "--jsonFile", required=False, help = "Path and name of the JSON file used to save/load the docs")


    # Read arguments from command line
    args = parser.parse_args()

    print(f'pdfPath: {args.pdfPath}')
    print(f'processType: {args.processType}')
    print(f'jsonFile: {args.jsonFile}')

    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if (args.pdfPath):
        pdf_files_path = args.pdfPath
    else :
        pdf_files_path = os.path.join(current_dir, "data/example_pdfs/")
    
    if (args.jsonFile):
        json_file = args.jsonFile
    else:
        json_file = os.path.join(current_dir, "temp/pdf_demo/parsed_pdf_docs.json")

    pdfDemo = PDFDemo()
    
    if args.processType == str_loadOnly:
        docs = pdfDemo.load_docs_from_json(json_file)
        print(f"Loaded {len(docs)} docs")
    elif args.processType == str_processAll:
        docs = pdfDemo.process_pdfs(pdf_files_path=pdf_files_path)
        pdfDemo.save_docs_as_json(docs, json_file)
        print(f"Saved {len(docs)} docs")
    else:
        print("Invalid process type")
        sys.exit(1)

if __name__ == "__main__":
    main()