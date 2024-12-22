import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

class Agents:
    def __init__(self):
        pass

    llm = ChatGoogleGenerativeAI(model="gemini-pro")
    french_template = PromptTemplate(
        input_variables=["text"],
        template="Translate the following English text to French: '{text}'"
    )

    german_template = PromptTemplate(
        input_variables=["text"],
        template="Translate the following English text to German: '{text}'"
    )

    ## Creating LLMChains for each translation
    french_chain = french_template | llm | StrOutputParser()
    german_chain = german_template | llm | StrOutputParser()


    def translate(self, text):
        french = self.french_chain.invoke(text)
        german = self.german_chain.invoke(text)
        return french, german


def main():
        text = input("Enter English text to translate: ")
        french, german = Agents().translate(text)
        print(f"French translation: {french}")
        print(f"German translation: {german}")


if __name__ == "__main__":
    main()
