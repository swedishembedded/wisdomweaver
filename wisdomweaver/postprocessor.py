import openai
import os

# Set api key
openai.api_key = os.getenv("OPENAI_API_KEY")


class Postprocessor:
    def __init__(self):
        pass

    def postprocess(self, query, result):
        message = """
When querying our database for query "%s", we got the following answers.

####
%s
####

Rewrite the above answers so that they answer the original query as accurately
as possible.
""" % (
            query,
            result,
        )
        print("=====")
        print(message)
        print("=====")
        self.stream(message, print)

    def stream(self, user_text, cb):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": user_text}],
            max_tokens=2000,
            stream=True,
            temperature=0.8,
        )

        for chunk in response:
            delta = chunk.choices[0]["delta"]
            if "content" in delta:
                cb(delta["content"], end="")
