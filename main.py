from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

def main():
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-5-nano")

    # Create a simple prompt
    prompt = ChatPromptTemplate.from_messages([
        ("human", "GIVE ME THE BEAST STRATEGY TO WIN ON HACKATHON ABOUT ai AGENT")
    ])

    # Run the chain
    chain = prompt | llm
    response = chain.invoke({})

    print("AI Response:\n", response.content)

if __name__ == "__main__":
    main()
