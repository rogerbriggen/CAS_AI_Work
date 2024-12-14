import base64
import io
from openai import OpenAI
import sys
import os

class OpenAIWrapper:
    def __init__(self):
        # Initializing OpenAI client - see https://platform.openai.com/docs/quickstart?context=python
        self.openai = OpenAI(api_key=os.getenv("MY_OPENAI_API_KEY"))
        self.system_prompt_image = '''
You will be provided with an image of a PDF page or a slide. Your goal is to deliver a detailed and engaging presentation about the content you see, using clear and accessible language suitable for a 101-level audience.

If there is an identifiable title, start by stating the title to provide context for your audience.

Describe visual elements in detail:

- **Diagrams**: Explain each component and how they interact. For example, "The process begins with X, which then leads to Y and results in Z."

- **Tables**: Break down the information logically. For instance, "Product A costs X dollars, while Product B is priced at Y dollars."

Focus on the content itself rather than the format:

- **DO NOT** include terms referring to the content format.

- **DO NOT** mention the content type. Instead, directly discuss the information presented.

Keep your explanation comprehensive yet concise:

- Be exhaustive in describing the content, as your audience cannot see the image.

- Exclude irrelevant details such as page numbers or the position of elements on the image.

Use clear and accessible language:

- Explain technical terms or concepts in simple language appropriate for a 101-level audience.

Engage with the content:

- Interpret and analyze the information where appropriate, offering insights to help the audience understand its significance.

------

If there is an identifiable title, present the output in the following format:

{TITLE}

{Content description}

If there is no clear title, simply provide the content description.
'''

    # Converting images to base64 encoded images in a data URI format to use with the ChatCompletions API
    def get_img_uri(self, img):
        png_buffer = io.BytesIO()
        img.save(png_buffer, format="PNG")
        png_buffer.seek(0)

        base64_png = base64.b64encode(png_buffer.read()).decode('utf-8')

        data_uri = f"data:image/png;base64,{base64_png}"
        return data_uri
    
    def get_img_uri_from_pngfile(self, path):
        with open(path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        data_uri = f"data:image/png;base64,{base64_image}"
        return data_uri
    
    def analyze_image(self, data_uri, system_prompt=None):
        if system_prompt is None:
            system_prompt = self.system_prompt_image
        response = self.openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                        "type": "image_url",
                        "image_url": {
                            "url": f"{data_uri}"
                        }
                        }
                    ]
                    },
            ],
            max_tokens=500, # maximum number of tokens to generate
            temperature=0, # 0 is deterministic, less random
            top_p=0.1 # use only the top 10% of the probability mass for the next token
        )
        return response.choices[0].message.content
            


def main():
    if (len(sys.argv) == 1):
        # we use our demo pdf if no pdf is provided
        current_dir = os.path.dirname(os.path.abspath(__file__))
        pdf_path = os.path.join(current_dir, "temp/images/pdf_preprocess/page_1.png")
    elif (len(sys.argv) == 2) and (sys.argv[1] == "--help"):
        print("Usage: python openai_wrapper.py  <image_path_and_name>")
        sys.exit(0)
    elif (len(sys.argv) == 2):
        pdf_path = sys.argv[1]
    else:
        print("Usage: python openai_wrapper.py <image_path_and_name>")
        sys.exit(1)

    wrapper = OpenAIWrapper()
    
    try:
        image_uri = wrapper.get_img_uri_from_pngfile(pdf_path)
        print(f"Converted image to base64 encoded image in data URI format.")
        text = wrapper.analyze_image(image_uri)
        print(f"Extracted text: {text[:500]}...")  # Print first 500 characters of extracted text

        # Save extracted text to temp directory for inspection
        text_output_dir = os.path.join(current_dir, "temp/text/openai_wrapper")
        os.makedirs(text_output_dir, exist_ok=True)
        text_path = os.path.join(text_output_dir, "extracted_text.txt")
        with open(text_path, "w", encoding="utf-8") as text_file:
            text_file.write(text)
        print(f"Saved extracted text to {text_path}.")

    except Exception as e:
        print(f"Error processing openai: {e}")

if __name__ == "__main__":
    main()