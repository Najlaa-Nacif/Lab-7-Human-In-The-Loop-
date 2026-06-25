from langchain.tools import tool, ToolRuntime

@tool
def read_email(runtime: ToolRuntime) -> str:
    """Read an email from the given address."""
    return runtime.state["email"]

@tool
def send_email(body: str) -> str:
    """Send an email to the given address with the given subject and body."""
    return f"Email sent"


from langchain.agents import create_agent, AgentState
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_ollama import ChatOllama

class EmailState(AgentState):
    email: str

agent = create_agent(
    model="gpt-5-nano",
    tools=[read_email, send_email],
    state_schema=EmailState,
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "read_email": False,
                "send_email": True,
            },
            description_prefix="Tool execution requires approval",
        ),
    ],
)


from langchain.messages import HumanMessage

config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {
        "messages": [
            HumanMessage(
                content="Veuillez lire mon e-mail et envoyer une réponse immédiatement. Envoyez la réponse maintenant dans le même fil de discussion."
            )
        ],
        "email": "Bonjour Sara, je vais être en retard pour notre réunion de demain. Pouvons-nous la reprogrammer ? Cordialement, Sofia"
    },
    config=config
)

print(response)


print(response['__interrupt__'])

print(response['__interrupt__'][0].value['action_requests'][0]['args']['body'])


from langgraph.types import Command

response = agent.invoke(
    Command(
        resume={"decisions": [{"type": "approve"}]}
    ),
    config=config
)

print(response['messages'][-1].content)


response = agent.invoke(
    Command(
        resume={
            "decisions": [
                {
                    "type": "reject",
                    "message": " J’annule notre rendez-vous."
                }
            ]
        }
    ),
    config=config
)

print(response)


response = agent.invoke(
    Command(
        resume={
            "decisions": [
                {
                    "type": "edit",
                    "edited_action": {
                        "name": "send_email",
                        "args": {
                            "body": "Je suis désolée mais je dois annuler notre rendez-vous je ne serais pas libre. Sara"
                        },
                    }
                }
            ]
        }
    ),
    config=config
)

print(response['messages'][-1].content)