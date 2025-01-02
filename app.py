from subgraph.graph_states import ResearcherState
from main_graph.graph_states import AgentState
from utils.utils import config

def main():
    state1 = ResearcherState(question="")
    state2 = AgentState(messages="")

    print(config["retriever"]["file"])

if __name__ == "__main__":
    main()

