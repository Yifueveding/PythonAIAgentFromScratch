import os

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import Optional


load_dotenv()
load_dotenv("sample.env", override=False)

for env_key in ("ANTHROPIC_API_KEY", "ANTHROPIC_MODEL"):
    if os.getenv(env_key):
        os.environ[env_key] = os.environ[env_key].strip()

from gps_cluster_tools import save_tool, search_tool


class GPSClusterResponse(BaseModel):
    date: str
    vehicle_number: int
    cluster: Optional[int]
    recommendation: str
    rationale: str
    similar_dates: list[str]
    tools_used: list[str]


llm = ChatAnthropic(model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"))
parser = PydanticOutputParser(pydantic_object=GPSClusterResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a GPS trajectory recommendation assistant.
            The user will ask for a date and may include a vehicle number. Use Vehicle
            689 by default unless the user asks for a different vehicle. Use the search
            tool to query the date and vehicle and find which GPS image cluster the
            trajectory belongs to. Then use the save tool to save the recommendation.

            Describe the result as a recommendation for how to categorize the upcoming
            GPS trajectory. Be concise and action-oriented. The recommendation should
            say which cluster to use, and the rationale should mention the nearest
            similar dates returned by the search tool.

            Return only the structured response requested by the format instructions.
            Do not add markdown, bullets, headings, emojis, or any extra summary after
            the structured response.
            \n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, save_tool]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools,
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
query = input("Which Vehicle 689 GPS trajectory date should I categorize? ")

try:
    raw_response = agent_executor.invoke({"query": query})
except Exception as e:
    print(f"Agent execution failed: {e}")
    raise SystemExit(1)

try:
    output = raw_response.get("output")
    if isinstance(output, list):
        output = "\n".join(
            block.get("text", str(block)) if isinstance(block, dict) else str(block)
            for block in output
        )
    structured_response = parser.parse(output)
    print(structured_response)
except Exception as e:
    print("Error parsing response", e, "Raw Response - ", raw_response)
