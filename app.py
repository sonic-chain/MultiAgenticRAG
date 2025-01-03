from subgraph.graph_states import ResearcherState
from main_graph.graph_states import AgentState
from utils.utils import config
from subgraph.graph_builder import researcher_graph
from main_graph.graph_builder import InputState, graph
from langgraph.types import Command
import asyncio
import uuid
#!/usr/bin/env python3

import asyncio
import time
import builtins
import sys

def new_uuid():
    return str(uuid.uuid4())
    
# Make sure you have these objects available from your code/environment:
# from your_module import graph, InputState, Command, new_uuid

# If your code for new_uuid or relevant imports is somewhere else, import or define them here:
# def new_uuid():
#     import uuid
#     return str(uuid.uuid4())

async def main():
    # Override the built-in input if needed
    input = builtins.input

    # Define your query
    query = "This is a question related to environmental context. tell me the data center PUE efficiency value in Dublin in 2022"

    # Build the input state
    inputState = InputState(messages=query)

    # Build or retrieve a unique thread/config
    thread = {"configurable": {"thread_id": new_uuid()}}  # make sure new_uuid() is defined

    # First streaming call
    #async for c in graph.astream(input=inputState, stream_mode="messages", config=thread):
    async for c, metadata in graph.astream(input=inputState, stream_mode="messages", config=thread):
        if c.content:
            time.sleep(0.1)
            print(c.content, end="", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
