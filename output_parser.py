from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser, PydanticOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model="claude-sonnet-4-6")

def templates(parser):
    template1 = PromptTemplate(
        template="Write a detail report on {topic} \n {format_instructions}",
        input_variables=['topic'],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    template2 = PromptTemplate(
        template="Write 5 line summary on the followin text. \n {text} \n {format_instructions}",
        input_variables=['text'],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    return template1, template2

str_parser = StrOutputParser()
json_parser = JsonOutputParser()

template1, template2 = templates(json_parser)


prompt = template1.invoke({
    'topic':'black hole'
})

result = model.invoke(prompt)
final_result = json_parser.parse(result.content)
print(final_result)



