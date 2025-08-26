#!/usr/bin/env python3
"""
Interactive graph visualization with planted subgraph detection using LRMC algorithm.
Generates random graphs with controllable planted subgraph density and provides
pygame-based visualization with pan/zoom capabilities.
"""

import pygame
import numpy as np
import networkx as nx
import random
import subprocess
import tempfile
import os
from typing import Tuple, List, Set, Optional
import math
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

class GraphVisualizer:
    def __init__(self, width=1200, height=800):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("LRMC Graph Visualization - Planted Subgraph Detection")

        # Rich console for terminal output
        self.console = Console()

        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GRAY = (128, 128, 128)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.YELLOW = (255, 255, 0)
        self.CYAN = (0, 255, 255)
        self.MAGENTA = (255, 0, 255)

        # Camera/View settings
        self.camera_x = width // 2
        self.camera_y = height // 2
        self.zoom = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 5.0

        # Graph data
        self.graph = None
        self.node_positions = {}
        self.planted_nodes = set()
        self.detected_component = set()

        # UI state
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        self.running = True
        self.dragging = False
        self.last_mouse_pos = None

        # Graph generation parameters
        self.n_nodes = 100
        self.subgraph_size = 20
        self.base_density = 0.05
        self.subgraph_density = 0.8
        self.epsilon = 1000

        self.clock = pygame.time.Clock()

        # Display initial settings
        self.display_settings()

    def clear_terminal(self):
        """Clear the terminal screen."""
        if os.name == 'posix':
            subprocess.run(['clear'])
        else:
            subprocess.run(['cls'], shell=True)

    def display_settings(self):
        """Display current settings using rich formatting."""
        self.clear_terminal()

        # Create a table for settings
        table = Table(title="🔧 Current Graph Settings", show_header=True, header_style="bold magenta")
        table.add_column("Setting", style="cyan", width=20)
        table.add_column("Value", style="yellow", width=15)
        table.add_column("Controls", style="green", width=20)
        table.add_column("Description", style="white")

        table.add_row(
            "Total Nodes",
            str(self.n_nodes),
            "1/2 keys",
            "Number of nodes in the graph"
        )
        table.add_row(
            "Subgraph Size",
            str(self.subgraph_size),
            "3/4 keys",
            "Size of the planted dense subgraph"
        )
        table.add_row(
            "Base Density",
            f"{self.base_density:.3f}",
            "5/6 keys",
            "Edge probability outside subgraph"
        )
        table.add_row(
            "Subgraph Density",
            f"{self.subgraph_density:.1f}",
            "7/8 keys",
            "Edge probability within subgraph"
        )

        # Create control instructions panel
        controls_text = Text()
        controls_text.append("🎮 Controls:\n", style="bold cyan")
        controls_text.append("SPACE", style="bold yellow")
        controls_text.append(" - Generate new graph\n")
        controls_text.append("R", style="bold yellow")
        controls_text.append(" - Run LRMC algorithm\n")
        controls_text.append("+/-", style="bold yellow")
        controls_text.append(" - Zoom in/out\n")
        controls_text.append("Mouse", style="bold yellow")
        controls_text.append(" - Drag to pan, wheel to zoom")

        controls_panel = Panel(controls_text, title="Instructions", border_style="blue")

        # Create color legend panel
        legend_text = Text()
        legend_text.append("🎨 Color Legend:\n", style="bold cyan")
        legend_text.append("● ", style="red")
        legend_text.append("Planted subgraph nodes/edges\n")
        legend_text.append("● ", style="green")
        legend_text.append("LRMC detected nodes\n")
        legend_text.append("● ", style="blue")
        legend_text.append("Regular nodes\n")
        legend_text.append("● ", style="yellow")
        legend_text.append("Subgraph boundary edges\n")
        legend_text.append("● ", style="white")
        legend_text.append("Outside edges")

        legend_panel = Panel(legend_text, title="Visualization", border_style="magenta")

        # Display everything
        self.console.print()
        self.console.print(table)
        self.console.print()

        # Display panels side by side
        from rich.columns import Columns
        self.console.print(Columns([controls_panel, legend_panel]))

        # Show current graph status
        if self.graph:
            status_text = Text()
            status_text.append("📊 Current Graph: ", style="bold white")
            status_text.append(f"{len(self.graph.nodes())} nodes, ", style="cyan")
            status_text.append(f"{len(self.graph.edges())} edges, ", style="cyan")
            status_text.append(f"{len(self.planted_nodes)} planted, ", style="red")
            status_text.append(f"{len(self.detected_component)} detected", style="green")

            status_panel = Panel(status_text, title="Graph Status", border_style="yellow")
            self.console.print()
            self.console.print(status_panel)

    def generate_random_graph(self, n_nodes: int, base_density: float,
                            subgraph_size: int, subgraph_density: float) -> nx.Graph:
        """Generate random graph with planted dense subgraph."""
        G = nx.Graph()
        G.add_nodes_from(range(n_nodes))

        # Select random subgraph nodes
        planted_nodes = set(random.sample(range(n_nodes), subgraph_size))

        # Add base edges (outside subgraph)
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if i not in planted_nodes and j not in planted_nodes:
                    if random.random() < base_density:
                        G.add_edge(i, j)

        # Add subgraph edges (higher density)
        subgraph_nodes = list(planted_nodes)
        for i in range(len(subgraph_nodes)):
            for j in range(i + 1, len(subgraph_nodes)):
                if random.random() < subgraph_density:
                    G.add_edge(subgraph_nodes[i], subgraph_nodes[j])

        # Add some edges between subgraph and outside
        for node in planted_nodes:
            for other in range(n_nodes):
                if other not in planted_nodes and random.random() < base_density * 0.5:
                    G.add_edge(node, other)

        return G, planted_nodes

    def layout_graph_circular(self, G: nx.Graph) -> dict:
        """Position nodes in a circle for better visualization."""
        positions = {}
        n = len(G.nodes())

        # Main circle
        radius = min(self.width, self.height) * 0.3

        for i, node in enumerate(G.nodes()):
            angle = 2 * math.pi * i / n
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            positions[node] = (x, y)

        return positions

    def world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates."""
        screen_x = int((x * self.zoom) + self.camera_x)
        screen_y = int((y * self.zoom) + self.camera_y)
        return screen_x, screen_y

    def screen_to_world(self, screen_x: int, screen_y: int) -> Tuple[float, float]:
        """Convert screen coordinates to world coordinates."""
        x = (screen_x - self.camera_x) / self.zoom
        y = (screen_y - self.camera_y) / self.zoom
        return x, y

    def draw_graph(self):
        """Draw the graph with current view settings."""
        self.screen.fill(self.BLACK)

        if not self.graph:
            # Draw instructions
            instructions = [
                "Press SPACE to generate new graph",
                "Press R to run LRMC algorithm",
                "Press +/- to adjust zoom",
                "Drag to pan, Mouse wheel to zoom",
                f"Current settings: N={self.n_nodes}, Subgraph={self.subgraph_size}, "
                f"Base density={self.base_density}, Subgraph density={self.subgraph_density}"
            ]

            y_offset = 50
            for instruction in instructions:
                text = self.small_font.render(instruction, True, self.WHITE)
                self.screen.blit(text, (10, y_offset))
                y_offset += 25
            return

        # Draw edges
        for edge in self.graph.edges():
            u, v = edge
            if u in self.node_positions and v in self.node_positions:
                x1, y1 = self.world_to_screen(*self.node_positions[u])
                x2, y2 = self.world_to_screen(*self.node_positions[v])

                # Color based on edge type
                if u in self.planted_nodes and v in self.planted_nodes:
                    color = self.RED  # Subgraph internal edge
                elif u in self.planted_nodes or v in self.planted_nodes:
                    color = self.YELLOW  # Edge connecting subgraph to outside
                else:
                    color = self.GRAY  # Outside edge

                pygame.draw.line(self.screen, color, (x1, y1), (x2, y2), 1)

        # Draw nodes
        for node in self.graph.nodes():
            if node in self.node_positions:
                x, y = self.world_to_screen(*self.node_positions[node])
                radius = max(3, int(8 * self.zoom))

                # Node color based on status
                if node in self.detected_component:
                    color = self.GREEN  # Detected by algorithm
                elif node in self.planted_nodes:
                    color = self.RED  # Planted subgraph
                else:
                    color = self.BLUE  # Regular node

                pygame.draw.circle(self.screen, color, (x, y), radius)

                # Draw node label if zoomed in enough
                if self.zoom > 0.5:
                    label = self.small_font.render(str(node), True, self.WHITE)
                    label_rect = label.get_rect(center=(x, y - radius - 10))
                    self.screen.blit(label, label_rect)

        # Draw legend
        self.draw_legend()

        # Draw stats
        self.draw_stats()

    def draw_legend(self):
        """Draw color legend."""
        legend_x = 10
        legend_y = 10

        legend_items = [
            ("Planted subgraph", self.RED),
            ("Detected component", self.GREEN),
            ("Regular nodes", self.BLUE),
            ("Subgraph edges", self.RED),
            ("Boundary edges", self.YELLOW),
            ("Outside edges", self.GRAY)
        ]

        for i, (label, color) in enumerate(legend_items):
            y = legend_y + i * 20
            pygame.draw.circle(self.screen, color, (legend_x + 10, y + 10), 8)
            text = self.small_font.render(label, True, self.WHITE)
            self.screen.blit(text, (legend_x + 25, y))

    def draw_stats(self):
        """Draw graph statistics."""
        stats_y = self.height - 120

        if self.graph:
            stats = [
                f"Nodes: {len(self.graph.nodes())}",
                f"Edges: {len(self.graph.edges())}",
                f"Planted nodes: {len(self.planted_nodes)}",
                f"Detected nodes: {len(self.detected_component)}",
                f"Zoom: {self.zoom:.2f}x"
            ]

            for i, stat in enumerate(stats):
                text = self.small_font.render(stat, True, self.WHITE)
                self.screen.blit(text, (10, stats_y + i * 20))

    def run_lrmc_algorithm(self):
        """Run the Java LRMC algorithm on current graph."""
        if not self.graph:
            return

        # Create temporary input file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            # Write graph in format: n m, then edges
            n = len(self.graph.nodes())
            m = len(self.graph.edges())
            f.write(f"{n} {m}\n")

            for edge in self.graph.edges():
                u, v = edge
                f.write(f"{u+1} {v+1}\n")  # Java uses 1-based indexing

            temp_file = f.name

        try:
            # Run Java algorithm
            result = subprocess.run(
                ['java', 'UsacoProbs.clique2_mk_benchmark_accuracy', str(self.epsilon), temp_file],
                capture_output=True,
                text=True,
                cwd=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
            )

            if result.returncode == 0:
                # Parse output to extract detected component
                self.detected_component.clear()

                for line in result.stdout.split('\n'):
                    if line.startswith('COMPONENT:'):
                        nodes = line.split()[1:]  # Skip 'COMPONENT:'
                        self.detected_component = {int(node) - 1 for node in nodes}  # Convert to 0-based
                        break

                # Update display after algorithm runs
                self.display_settings()
            else:
                print(f"Algorithm failed: {result.stderr}")

        except Exception as e:
            print(f"Error running algorithm: {e}")
        finally:
            # Clean up temporary file
            os.unlink(temp_file)

    def handle_events(self):
        """Handle pygame events."""
        settings_changed = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # Generate new graph
                    self.graph, self.planted_nodes = self.generate_random_graph(
                        self.n_nodes, self.base_density,
                        self.subgraph_size, self.subgraph_density
                    )
                    self.node_positions = self.layout_graph_circular(self.graph)
                    self.detected_component.clear()
                    self.display_settings()

                elif event.key == pygame.K_r:
                    # Run LRMC algorithm
                    self.run_lrmc_algorithm()

                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    # Zoom in
                    self.zoom = min(self.zoom * 1.2, self.max_zoom)

                elif event.key == pygame.K_MINUS:
                    # Zoom out
                    self.zoom = max(self.zoom / 1.2, self.min_zoom)

                elif event.key == pygame.K_1:
                    # Adjust parameters
                    self.n_nodes = max(10, self.n_nodes - 10)
                    settings_changed = True

                elif event.key == pygame.K_2:
                    self.n_nodes = min(500, self.n_nodes + 10)
                    settings_changed = True

                elif event.key == pygame.K_3:
                    self.subgraph_size = max(5, self.subgraph_size - 5)
                    settings_changed = True

                elif event.key == pygame.K_4:
                    self.subgraph_size = min(self.n_nodes // 2, self.subgraph_size + 5)
                    settings_changed = True

                elif event.key == pygame.K_5:
                    self.base_density = max(0.01, self.base_density - 0.05)
                    settings_changed = True

                elif event.key == pygame.K_6:
                    self.base_density = min(0.5, self.base_density + 0.05)
                    settings_changed = True

                elif event.key == pygame.K_7:
                    self.subgraph_density = max(0.1, self.subgraph_density - 0.1)
                    settings_changed = True

                elif event.key == pygame.K_8:
                    self.subgraph_density = min(1.0, self.subgraph_density + 0.1)
                    settings_changed = True

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    self.dragging = True
                    self.last_mouse_pos = pygame.mouse.get_pos()
                elif event.button == 4:  # Mouse wheel up
                    self.zoom = min(self.zoom * 1.1, self.max_zoom)
                elif event.button == 5:  # Mouse wheel down
                    self.zoom = max(self.zoom / 1.1, self.min_zoom)

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.dragging = False

            elif event.type == pygame.MOUSEMOTION:
                if self.dragging and self.last_mouse_pos:
                    current_pos = pygame.mouse.get_pos()
                    dx = current_pos[0] - self.last_mouse_pos[0]
                    dy = current_pos[1] - self.last_mouse_pos[1]
                    self.camera_x += dx
                    self.camera_y += dy
                    self.last_mouse_pos = current_pos

        # Update display if settings changed
        if settings_changed:
            self.display_settings()

    def run(self):
        """Main visualization loop."""
        while self.running:
            self.handle_events()
            self.draw_graph()
            pygame.display.flip()
            self.clock.tick(60)  # 60 FPS

        pygame.quit()

def main():
    """Main function to run the visualizer."""
    visualizer = GraphVisualizer()
    visualizer.run()

if __name__ == "__main__":
    main()
