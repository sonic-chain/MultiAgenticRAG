from subgraph.graph_states import ResearcherState
from main_graph.graph_states import AgentState

def main():
    state1 = ResearcherState(question="")
    state2 = AgentState(messages="")

    print(state2.hallucination)

if __name__ == "__main__":
    main()

