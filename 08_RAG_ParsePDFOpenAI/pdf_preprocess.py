from pdf2image import convert_from_path
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)
from pdfminer.high_level import extract_text
import sys
import os

class PDFPreprocess:
    

    def convert_to_images(self, path):
        poppler_path = os.getenv('POPPLER_PATH', 'c:/python_stuff/poppler/Library/bin')
        images = convert_from_path(path, poppler_path=poppler_path)
        return images
    
    def extract_text_from_doc(self, path):
        text = extract_text(path)
        return text
    

def main():
    if (len(sys.argv) == 1):
        # we use our demo pdf if no pdf is provided
        current_dir = os.path.dirname(os.path.abspath(__file__))
        pdf_path = os.path.join(current_dir, "data/example_pdfs/evals-decks.pdf")
    elif (len(sys.argv) == 2) and (sys.argv[1] == "--help"):
        print("Usage: python pdf_preprocess_image.py <pdf_path>")
        sys.exit(0)
    elif (len(sys.argv) == 2):
        pdf_path = sys.argv[1]
    else:
        print("Usage: python pdf_preprocess_image.py <pdf_path>")
        sys.exit(1)

    processor = PDFPreprocess()
    
    try:
        images = processor.convert_to_images(pdf_path)
        print(f"Converted {len(images)} pages to images.")
        # Save images to temp directory for inspection
        output_dir = os.path.join(current_dir, "temp/images/pdf_preprocess")
        os.makedirs(output_dir, exist_ok=True)
        for i, image in enumerate(images):
            image_path = os.path.join(output_dir, f"page_{i + 1}.png")
            image.save(image_path, "PNG")
        print(f"Saved images to {output_dir}.")
        
        text = processor.extract_text_from_doc(pdf_path)
        print(f"Extracted text: {text[:500]}...")  # Print first 500 characters of extracted text
        # Save extracted text to temp directory for inspection
        text_output_dir = os.path.join(current_dir, "temp/text/pdf_preprocess")
        os.makedirs(text_output_dir, exist_ok=True)
        text_path = os.path.join(text_output_dir, "extracted_text.txt")
        with open(text_path, "w", encoding="utf-8") as text_file:
            text_file.write(text)
        print(f"Saved extracted text to {text_path}.")

    except (PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError) as e:
        print(f"Error processing PDF: {e}")

if __name__ == "__main__":
    main()