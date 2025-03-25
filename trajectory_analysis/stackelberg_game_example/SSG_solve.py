import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import nashpy as nash

class StackelbergTreeViz:

    def __init__(self, leader_payoffs, follower_payoffs):
        self.leader_payoffs = np.array(leader_payoffs)
        self.follower_payoffs = np.array(follower_payoffs)
        self.G = nx.DiGraph()
        self.pos = {}
        self.node_colors = []
        self.edge_colors = []
        self.edge_labels = {}
        
    def create_tree(self):

        num_leader_strategies = len(self.leader_payoffs)
        num_follower_strategies = len(self.leader_payoffs[0])

        self.G.add_node("L")
        self.pos["L"] = (0.5, 1)
        self.node_colors.append('lightblue')
        
        leader_spacing = 1.0 / (num_leader_strategies + 1)
        follower_spacing = leader_spacing / (num_follower_strategies + 1)

        for i in range(len(self.leader_payoffs)):
            follower_node = f"F{i}"
            self.G.add_node(follower_node)
            self.G.add_edge("L", follower_node)
            self.pos[follower_node] = (leader_spacing * (i + 1), 0.7)
            self.node_colors.append('orange')
            self.edge_colors.append('gray')
            self.edge_labels[("L", follower_node)] = f"s{i+1}"
            
            for j in range(len(self.leader_payoffs[0])):
                terminal_node = f"T{i}{j}"
                self.G.add_node(terminal_node)
                self.G.add_edge(follower_node, terminal_node)
                self.pos[terminal_node] = (leader_spacing * (i + 1) + follower_spacing * (j - num_follower_strategies/2), 0.3)
                self.node_colors.append('lightgreen')
                self.edge_colors.append('gray')
                self.edge_labels[(follower_node, terminal_node)] = f"a{j+1}"

    def solve_game(self):
        num_leader_strategies = len(self.leader_payoffs)
        follower_best_responses = {}
        
        # Find follower's best responses for each leader strategy
        for leader_strategy in range(num_leader_strategies):
            follower_payoffs = self.follower_payoffs[leader_strategy]
            best_response = np.argmax(follower_payoffs)
            print(f"Leader strategy {leader_strategy}: Follower's best response is {best_response}")
            follower_best_responses[leader_strategy] = best_response
        
        leader_values = [self.leader_payoffs[i][follower_best_responses[i]] 
                        for i in range(num_leader_strategies)]
        optimal_leader_strategy = np.argmax(leader_values)
        optimal_follower_response = follower_best_responses[optimal_leader_strategy]
        
        optimal_path = [
            ("L", f"F{optimal_leader_strategy}"),
            (f"F{optimal_leader_strategy}", f"T{optimal_leader_strategy}{optimal_follower_response}")
        ]

        print(f"Optimal path: {optimal_path}")
        return optimal_path
    

    def draw_tree(self, highlight_solution=True, name="v1", output_path=None):
        self.create_tree()
        plt.figure(figsize=(30, 4))
        
        nx.draw_networkx_nodes(self.G, self.pos, node_color=self.node_colors, node_size=1000)
        
        if highlight_solution:
            optimal_path = self.solve_game()
            edge_colors = ['red' if edge in optimal_path else 'gray' for edge in self.G.edges()]
        else:
            edge_colors = ['gray'] * len(self.G.edges())
            
        nx.draw_networkx_edges(self.G, self.pos, edge_color=edge_colors, width=2)
        nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels=self.edge_labels)

        labels = {}
        for node in self.G.nodes():
            if node == "L":
                labels[node] = "Leader"
            elif node.startswith("F"):
                labels[node] = f"Follower\n(L:{node[1]})"
            else:
                i, j = int(node[1]), int(node[2])
                labels[node] = f"({self.leader_payoffs[i][j]},{self.follower_payoffs[i][j]})"
        
        nx.draw_networkx_labels(self.G, self.pos, labels)
        plt.axis('off')

        if output_path == None:
            plt.savefig("trajectory_analysis/stackelberg_game_example/game_tree_" + name + ".png", dpi=300, bbox_inches='tight')
        else:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')

        plt.close()

if __name__ == "__main__":

    leader_payoffs = [
        [3, 2, 5],  
        [2, 1, 2],  
        [1, 3, 2]   
    ]
    
    follower_payoffs = [
        [1, 4, 3],  
        [3, 1, 2], 
        [2, 2, 1]   
    ]

    viz = StackelbergTreeViz(leader_payoffs, follower_payoffs)
    viz.draw_tree(highlight_solution=True, name="v1")

    print("\n")

    viz = StackelbergTreeViz(follower_payoffs, leader_payoffs)
    viz.draw_tree(highlight_solution=True, name="v2")

    print("\n")

    A = np.array(leader_payoffs) 
    B = np.array(follower_payoffs) 

    game2 = nash.Game(A,B)

    print("Game Details:", game2)

    equilibria = game2.vertex_enumeration()

    for eq in equilibria:
        print("Equilibrium Strategy:", eq)

    print("\n")

    A = np.array(follower_payoffs) 
    B = np.array(leader_payoffs) 

    game2 = nash.Game(A,B)

    print("Game Details:", game2)

    equilibria = game2.vertex_enumeration()

    for eq in equilibria:
        print("Equilibrium Strategy:", eq)