#!/usr/bin/env python3

import argparse
import sys
import pygame
import math
from collections import defaultdict

# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 1000, 800
BACKGROUND_COLOR = (255, 255, 255)
NODE_COLOR = (0, 0, 255)
CLIQUE_NODE_COLOR = (255, 0, 0)
EDGE_COLOR = (200, 200, 200)
NODE_RADIUS = 8
CLIQUE_NODE_RADIUS = 10
FONT_SIZE = 16

class GraphVisualizer:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Graph Visualizer")
        self.font = pygame.font.Font(None, FONT_SIZE)
        self.clock = pygame.time.Clock()

        # Graph data
        self.nodes = {}
        self.edges = []
        self.clique_nodes = set()

        # Visualization state
        self.offset = [0, 0]  # x, y offset for panning
        self.scale = 1.0
        self.dragging = False
        self.last_mouse_pos = (0, 0)

    def load_graph(self, graph_file):
        """Load graph from file"""
        with open(graph_file, 'r') as f:
            # Read first line: N M
            line = f.readline().strip()
            if not line:
                raise ValueError("Empty graph file")

            parts = line.split()
            if len(parts) != 2:
                raise ValueError("Invalid graph file format. First line should be 'N M'")

            n, m = int(parts[0]), int(parts[1])

            # Read edges
            edges = []
            for _ in range(m):
                line = f.readline().strip()
                if not line:
                    break
                parts = line.split()
                if len(parts) != 2:
                    raise ValueError(f"Invalid edge format: {line}")
                u, v = int(parts[0]), int(parts[1])
                edges.append((u, v))

        self.edges = edges
        self._generate_layout()

    def load_clique_nodes(self, java_output_file):
        """Load clique nodes from Java output file"""
        # clique_nodes = set()

        # L-RMC component (convert from 1-indexed Java IDs back to PyG indices)
        lrmc_nodes_java = "1680 2883 901 617 1578 1103"

        # Load the saved remap dictionary from the benchmark script
        import pickle
        remap_filename = f"{java_output_file.lower()}_remap.pkl"
        with open(remap_filename, "rb") as f:
            remap_data = pickle.load(f)
        nodes_sorted = remap_data['nodes_sorted']

        # Convert Java 1-indexed IDs back to PyG indices
        lrmc_nodes_java_ids = [int(x) for x in lrmc_nodes_java.split()]
        lrmc_nodes = [nodes_sorted[j-1] for j in lrmc_nodes_java_ids]

        self.clique_nodes = lrmc_nodes

        # try:
        #     with open(java_output_file, 'r') as f:
        #         for line in f:
        #             if line.startswith("COMPONENT:"):
        #                 # Extract node numbers after "COMPONENT:"
        #                 node_strs = line.strip().split()[1:]  # Skip "COMPONENT:"
        #                 clique_nodes = set(int(node_str) for node_str in node_strs)
        #                 break
        # except FileNotFoundError:
        #     print(f"Warning: Java output file '{java_output_file}' not found. No clique nodes will be highlighted.")
        # except Exception as e:
        #     print(f"Warning: Could not parse clique nodes from '{java_output_file}': {e}")

        # self.clique_nodes = clique_nodes

    def _generate_layout(self):
        """Generate a simple force-directed layout for nodes"""
        # For simplicity, we'll use a circular layout for now
        # In a more advanced version, we could implement a force-directed algorithm
        n = len(set([u for u, v in self.edges] + [v for u, v in self.edges]))

        # Create a mapping from node IDs to indices
        all_nodes = set()
        for u, v in self.edges:
            all_nodes.add(u)
            all_nodes.add(v)
        all_nodes = list(all_nodes)
        all_nodes.sort()

        self.nodes = {}
        center_x, center_y = WIDTH // 2, HEIGHT // 2
        radius = min(WIDTH, HEIGHT) // 3

        for i, node_id in enumerate(all_nodes):
            angle = 2 * math.pi * i / len(all_nodes)
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            self.nodes[node_id] = (x, y)

    def _world_to_screen(self, x, y):
        """Convert world coordinates to screen coordinates"""
        return (x * self.scale + self.offset[0], y * self.scale + self.offset[1])

    def _screen_to_world(self, x, y):
        """Convert screen coordinates to world coordinates"""
        return ((x - self.offset[0]) / self.scale, (y - self.offset[1]) / self.scale)

    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    self.dragging = True
                    self.last_mouse_pos = event.pos
                elif event.button == 4:  # Mouse wheel up
                    # Zoom in
                    mouse_x, mouse_y = event.pos
                    world_x, world_y = self._screen_to_world(mouse_x, mouse_y)

                    self.scale *= 1.1

                    # Adjust offset to zoom towards mouse position
                    new_screen_x, new_screen_y = self._world_to_screen(world_x, world_y)
                    self.offset[0] += mouse_x - new_screen_x
                    self.offset[1] += mouse_y - new_screen_y
                elif event.button == 5:  # Mouse wheel down
                    # Zoom out
                    mouse_x, mouse_y = event.pos
                    world_x, world_y = self._screen_to_world(mouse_x, mouse_y)

                    self.scale *= 0.9

                    # Adjust offset to zoom towards mouse position
                    new_screen_x, new_screen_y = self._world_to_screen(world_x, world_y)
                    self.offset[0] += mouse_x - new_screen_x
                    self.offset[1] += mouse_y - new_screen_y

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left mouse button
                    self.dragging = False

            elif event.type == pygame.MOUSEMOTION:
                if self.dragging:
                    dx = event.pos[0] - self.last_mouse_pos[0]
                    dy = event.pos[1] - self.last_mouse_pos[1]
                    self.offset[0] += dx
                    self.offset[1] += dy
                    self.last_mouse_pos = event.pos

        return True

    def draw(self):
        """Draw the graph"""
        self.screen.fill(BACKGROUND_COLOR)

        # Draw edges
        for u, v in self.edges:
            if u in self.nodes and v in self.nodes:
                x1, y1 = self._world_to_screen(*self.nodes[u])
                x2, y2 = self._world_to_screen(*self.nodes[v])
                pygame.draw.line(self.screen, EDGE_COLOR, (x1, y1), (x2, y2), 1)

        # Draw nodes
        for node_id, (x, y) in self.nodes.items():
            screen_x, screen_y = self._world_to_screen(x, y)

            # Check if node is in clique
            color = CLIQUE_NODE_COLOR if node_id in self.clique_nodes else NODE_COLOR
            radius = CLIQUE_NODE_RADIUS if node_id in self.clique_nodes else NODE_RADIUS

            # Draw node
            pygame.draw.circle(self.screen, color, (int(screen_x), int(screen_y)), radius)

            # Draw node label for clique nodes or when zoomed in
            if node_id in self.clique_nodes or self.scale > 2.0:
                text = self.font.render(str(node_id), True, (0, 0, 0))
                self.screen.blit(text, (int(screen_x) + radius, int(screen_y) - radius))

        # Draw instructions
        instructions = [
            "Instructions:",
            "• Mouse wheel: Zoom in/out",
            "• Left click + drag: Pan",
            "• Blue nodes: Regular nodes",
            "• Red nodes: Clique nodes"
        ]

        for i, instruction in enumerate(instructions):
            text = self.font.render(instruction, True, (0, 0, 0))
            self.screen.blit(text, (10, 10 + i * 20))

        pygame.display.flip()

    def run(self):
        """Main loop"""
        running = True
        while running:
            running = self.handle_events()
            self.draw()
            self.clock.tick(60)  # 60 FPS

        pygame.quit()

def main():
    parser = argparse.ArgumentParser(description='Visualize graphs with clique nodes highlighted')
    parser.add_argument('graph_file', help='Path to the graph file')
    parser.add_argument('--clique-file', help='Path to the clique2_mk_benchmark_accuracy.java output file')

    args = parser.parse_args()

    # Create visualizer
    visualizer = GraphVisualizer()

    # Load graph
    try:
        visualizer.load_graph(args.graph_file)
    except Exception as e:
        print(f"Error loading graph file: {e}")
        sys.exit(1)

    # Load clique nodes if provided
    if args.clique_file:
        visualizer.load_clique_nodes(args.clique_file)

    # Run visualization
    visualizer.run()

if __name__ == "__main__":
    main()
