from openai import OpenAI
from decouple import config


client = OpenAI(api_key=config('API_KEY'))

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "Given a block of text, generate three queries based on the content of the text. These queries will be used for pseudo-labeling of a retrieval model's training data using sentence-BERT. The goal is to create relevant and diverse queries that capture the essence of the text and can be used to retrieve similar information."},
    {"role": "user", "content": "Please provide three queries based on the following text. These queries will be used to improve the performance of a retrieval model through pseudo-labeling with sentence-BERT. The text is as follows: I bought a Sony Ericsson X8 and saw on YouTube that when we turn on this phone a page is supposed to come up and mention the name of Sony Ericsson. When I turn on my phone just it says Android and that's all. When I wanted to hard restart it or upgrade it, I pressed the back button but nothing happens. When I hold left and right keys together, the same situation when I release. Is my phone a fake? Why it is like that? Your Task: Generate 3 queries that effectively summarize the content of the text and can be used to retrieve similar information. Consider the key concepts, entities, and context provided in the text to craft diverse and relevant queries. Each query should be distinct and capture different aspects of the text. Output Format:Generated queries should be separated by new line, there shouldn't be any output other than generated queries."}
  ]
)

print(completion.choices[0].message.content.split("\n"))