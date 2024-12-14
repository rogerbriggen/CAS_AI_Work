import os
from openai import OpenAI, OpenAIError, RateLimitError
import threading
import backoff # for exponential backoff


# Rate-limited openai API wrapper... implemented as singleton so that we can use the same instance across multiple calls and using
# https://pypi.org/project/backoff/ to handle rate limiting.
# A more sophisticated implementation would be here: https://github.com/openai/openai-cookbook/blob/main/examples/api_request_parallel_processor.py
# see also https://cookbook.openai.com/examples/how_to_handle_rate_limits


class OpenAIClientRL:
    _instance = None
    _lock = threading.Lock()

    # Singleton implementation, override __new__ method
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(OpenAIClientRL, cls).__new__(cls, *args, **kwargs)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        # Initializing OpenAI client - see https://platform.openai.com/docs/quickstart?context=python
        self.openai = OpenAI(api_key=os.getenv("MY_OPENAI_API_KEY"))

   # Exponential backoff decorator to handle rate limiting
    @backoff.on_exception(backoff.expo, RateLimitError)
    def chat_completions_create(self, **kwargs):
        return self.openai.chat.completions.create(**kwargs)




def main():
    # Usage
    client = OpenAIClientRL()
    response = client.chat_completions_create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": "Write a joke about programming or programmers.",
                    },
                ],
                max_tokens=500, # maximum number of tokens to generate
                temperature=0, # 0 is deterministic, less random
                top_p=0.1 # use only the top 10% of the probability mass for the next token)
    )
    print(f'\nresponse (raw): {response}')
    print(f'\nresponse (text): {response.choices[0].message.content}')

    client2 = OpenAIClientRL()
    print(f'client1 is client2: {client is client2}') # Check if they are the same instance... which they should since it should be singleton

if __name__ == "__main__":
    main()