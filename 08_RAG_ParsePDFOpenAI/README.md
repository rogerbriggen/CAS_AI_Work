# This example shows how to use RAG with OpenAI

The pdfs are from <https://github.com/openai/openai-cookbook/tree/main/examples/data/example_pdfs>.

## Environment

The conda environment is named openai.

The environment can be created with conda:

````shell
conda env create -f environment.yml
````

To save the updated environment:

````shell
conda env export > environment.yml
````

### Additional Software

We need poppler for pdf processing with pdf2image. See <https://pypi.org/project/pdf2image/>

### Environment variables

To do it the conda way, see <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#windows>

#### Poppler path

Set the path to the bin folder of poppler 
poppler_path = r"C:\path\to\poppler-xx\bin"

#### OpenAI API key

Set the environment variable *MY_OPENAI_API_KEY* to your api key.
