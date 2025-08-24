package UsacoProbs;

import java.io.*;
import java.util.*;
import java.util.stream.Collectors;

public class Table1Synthetic {

    static class SyntheticGraph {
        List<Integer>[] adj;
        Set<Integer> plantedCluster;
        int n, m;

        @SuppressWarnings("unchecked")
        SyntheticGraph(int n) {
            this.n = n;
            this.adj = new ArrayList[n + 1];
            for (int i = 1; i <= n; i++) {
                adj[i] = new ArrayList<>();
            }
            this.plantedCluster = new HashSet<>();
            this.m = 0;
        }

        void addEdge(int u, int v) {
            if (u != v && !adj[u].contains(v)) {
                adj[u].add(v);
                adj[v].add(u);
                m++;
            }
        }
    }

    static class EvaluationResult {
        Set<Integer> foundSubgraph;
        double precision, recall, f1;
        double density;
        int rmcScore;
        long runtimeMs;

        EvaluationResult(Set<Integer> found, Set<Integer> planted, long runtime) {
            this.foundSubgraph = found;
            this.runtimeMs = runtime;

            if (found.isEmpty()) {
                precision = recall = f1 = 0.0;
            } else {
                Set<Integer> intersection = new HashSet<>(found);
                intersection.retainAll(planted);

                precision = (double) intersection.size() / found.size();
                recall = planted.isEmpty() ? 0.0 : (double) intersection.size() / planted.size();
                f1 = (precision + recall == 0) ? 0.0 : 2 * precision * recall / (precision + recall);
            }
        }

        void computeDensity(List<Integer>[] adj) {
            if (foundSubgraph.size() <= 1) {
                density = 0.0;
                rmcScore = 0;
                return;
            }

            int edges = 0;
            int minDegree = Integer.MAX_VALUE;

            for (int u : foundSubgraph) {
                int internalDegree = 0;
                for (int v : adj[u]) {
                    if (foundSubgraph.contains(v)) {
                        if (u < v) edges++; // count each edge once
                        internalDegree++;
                    }
                }
                minDegree = Math.min(minDegree, internalDegree);
            }

            density = foundSubgraph.size() <= 1 ? 0.0 : 2.0 * edges / (foundSubgraph.size() * (foundSubgraph.size() - 1));
            rmcScore = foundSubgraph.size() * minDegree;
        }
    }

    // Generate synthetic graph with planted dense subgraph
    static SyntheticGraph generateGraph(int nTotal, int clusterSize, double pIn, double pOut, double pBackground, Random rand) {
        SyntheticGraph g = new SyntheticGraph(nTotal);

        // Plant dense cluster in nodes 1..clusterSize
        for (int i = 1; i <= clusterSize; i++) {
            g.plantedCluster.add(i);
        }

        // Add edges within planted cluster
        for (int i = 1; i <= clusterSize; i++) {
            for (int j = i + 1; j <= clusterSize; j++) {
                if (rand.nextDouble() < pIn) {
                    g.addEdge(i, j);
                }
            }
        }

        // Add edges between cluster and rest
        for (int i = 1; i <= clusterSize; i++) {
            for (int j = clusterSize + 1; j <= nTotal; j++) {
                if (rand.nextDouble() < pOut) {
                    g.addEdge(i, j);
                }
            }
        }

        // Add background edges
        for (int i = clusterSize + 1; i <= nTotal; i++) {
            for (int j = i + 1; j <= nTotal; j++) {
                if (rand.nextDouble() < pBackground) {
                    g.addEdge(i, j);
                }
            }
        }

        return g;
    }

    // Baseline 1: k-core with best k
    static EvaluationResult runKCore(SyntheticGraph g) {
        long start = System.nanoTime();

        int[] deg = new int[g.n + 1];
        for (int i = 1; i <= g.n; i++) {
            deg[i] = g.adj[i].size();
        }

        // Find the highest-k non-empty core (degeneracy core)
        int maxK = 0;
        for (int d : deg) maxK = Math.max(maxK, d);

        Set<Integer> bestCore = new HashSet<>();
        int bestK = 0;

        for (int k = maxK; k >= 1; k--) {
            Set<Integer> core = computeKCore(g, k);
            if (!core.isEmpty()) {
                bestCore = core;
                bestK = k;
                break; // Found the highest non-empty core
            }
        }

        long end = System.nanoTime();
        EvaluationResult result = new EvaluationResult(bestCore, g.plantedCluster, (end - start) / 1_000_000);
        result.computeDensity(g.adj);
        return result;
    }

    static Set<Integer> computeKCore(SyntheticGraph g, int k) {
        int[] deg = new int[g.n + 1];
        boolean[] removed = new boolean[g.n + 1];
        Queue<Integer> queue = new LinkedList<>();

        for (int i = 1; i <= g.n; i++) {
            deg[i] = g.adj[i].size();
            if (deg[i] < k) {
                queue.offer(i);
                removed[i] = true;
            }
        }

        while (!queue.isEmpty()) {
            int u = queue.poll();
            for (int v : g.adj[u]) {
                if (!removed[v]) {
                    deg[v]--;
                    if (deg[v] < k) {
                        queue.offer(v);
                        removed[v] = true;
                    }
                }
            }
        }

        Set<Integer> core = new HashSet<>();
        for (int i = 1; i <= g.n; i++) {
            if (!removed[i]) {
                core.add(i);
            }
        }
        return core;
    }

    // Baseline 2: Densest subgraph (greedy approximation)
    static EvaluationResult runDensestSubgraph(SyntheticGraph g) {
        long start = System.nanoTime();

        Set<Integer> remaining = new HashSet<>();
        for (int i = 1; i <= g.n; i++) {
            remaining.add(i);
        }

        Set<Integer> bestSubgraph = new HashSet<>();
        double bestDensity = 0.0;

        while (!remaining.isEmpty()) {
            double currentDensity = computeSubgraphDensity(g, remaining);
            if (currentDensity > bestDensity) {
                bestDensity = currentDensity;
                bestSubgraph = new HashSet<>(remaining);
            }

            // Remove node with minimum degree in current subgraph
            int minDegNode = -1;
            int minDeg = Integer.MAX_VALUE;
            for (int u : remaining) {
                int deg = 0;
                for (int v : g.adj[u]) {
                    if (remaining.contains(v)) deg++;
                }
                if (deg < minDeg) {
                    minDeg = deg;
                    minDegNode = u;
                }
            }

            if (minDegNode != -1) {
                remaining.remove(minDegNode);
            } else {
                break;
            }
        }

        long end = System.nanoTime();
        EvaluationResult result = new EvaluationResult(bestSubgraph, g.plantedCluster, (end - start) / 1_000_000);
        result.computeDensity(g.adj);
        return result;
    }

    static double computeSubgraphDensity(SyntheticGraph g, Set<Integer> nodes) {
        if (nodes.size() <= 1) return 0.0;

        int edges = 0;
        for (int u : nodes) {
            for (int v : g.adj[u]) {
                if (nodes.contains(v) && u < v) {
                    edges++;
                }
            }
        }

        return 2.0 * edges / nodes.size();
    }

    // Baseline 3: Greedy quasi-clique
    static EvaluationResult runQuasiClique(SyntheticGraph g) {
        long start = System.nanoTime();

        Set<Integer> bestClique = new HashSet<>();
        double bestScore = 0.0;

        // Try starting from each node
        for (int seed = 1; seed <= g.n; seed++) {
            Set<Integer> clique = growQuasiClique(g, seed, 0.7); // 70% density threshold
            double score = clique.size() * computeSubgraphDensity(g, clique);

            if (score > bestScore) {
                bestScore = score;
                bestClique = new HashSet<>(clique);
            }
        }

        long end = System.nanoTime();
        EvaluationResult result = new EvaluationResult(bestClique, g.plantedCluster, (end - start) / 1_000_000);
        result.computeDensity(g.adj);
        return result;
    }

    static Set<Integer> growQuasiClique(SyntheticGraph g, int seed, double minDensity) {
        Set<Integer> clique = new HashSet<>();
        clique.add(seed);

        Set<Integer> candidates = new HashSet<>(g.adj[seed]);

        while (!candidates.isEmpty()) {
            int bestCandidate = -1;
            double bestNewDensity = 0.0;

            for (int candidate : candidates) {
                Set<Integer> testClique = new HashSet<>(clique);
                testClique.add(candidate);
                double density = computeSubgraphDensity(g, testClique);

                if (density >= minDensity && density > bestNewDensity) {
                    bestNewDensity = density;
                    bestCandidate = candidate;
                }
            }

            if (bestCandidate == -1) break;

            clique.add(bestCandidate);
            candidates.remove(bestCandidate);

            // Update candidates: add neighbors of new node, remove non-candidates
            for (int neighbor : g.adj[bestCandidate]) {
                if (!clique.contains(neighbor)) {
                    candidates.add(neighbor);
                }
            }
        }

        return clique;
    }

    // Run L-RMC using your existing code
    static EvaluationResult runLRMC(SyntheticGraph g, double eps) {
        long start = System.nanoTime();

        // Pass g.n as an argument
        clique2_mk_benchmark_accuracy.Result result = clique2_mk_benchmark_accuracy.runLaplacianRMC(g.adj, g.n, eps);

        long end = System.nanoTime();

        // The result now directly contains the best component found
        Set<Integer> component = result.bestComponent;

        EvaluationResult evalResult = new EvaluationResult(component, g.plantedCluster, (end - start) / 1_000_000);
        evalResult.computeDensity(g.adj);
        return evalResult;
    }

    static Set<Integer> extractComponent(SyntheticGraph g, int root) {
        Set<Integer> component = new HashSet<>();
        Queue<Integer> queue = new LinkedList<>();
        boolean[] visited = new boolean[g.n + 1];

        queue.offer(root);
        visited[root] = true;
        component.add(root);

        while (!queue.isEmpty()) {
            int u = queue.poll();
            for (int v : g.adj[u]) {
                if (!visited[v]) {
                    visited[v] = true;
                    component.add(v);
                    queue.offer(v);
                }
            }
        }

        return component;
    }

    public static void main(String[] args) {
        Random rand = new Random(42); // Fixed seed for reproducibility

        // Experiment parameters
        int[] clusterSizes = {20, 50, 100};
        double[] internalDensities = {0.6, 0.8, 0.9};
        double[] externalDensities = {0.01, 0.03, 0.05, 0.07, 0.1, 0.15};
        int nTotal = 2000;
        double pBackground = 0.002;
        double eps = 500;

        System.out.println("Method,ClusterSize,InternalDensity,ExternalDensity,Precision,Recall,F1,Density,RMCScore,Runtime");

        for (int clusterSize : clusterSizes) {
            for (double pIn : internalDensities) {
                for (double pOut : externalDensities) {
                    // Generate graph
                    SyntheticGraph g = generateGraph(nTotal, clusterSize, pIn, pOut, pBackground, rand);

                    // Run all algorithms
                    EvaluationResult lrmcResult = runLRMC(g, eps);
                    EvaluationResult kcoreResult = runKCore(g);
                    EvaluationResult densestResult = runDensestSubgraph(g);
                    // EvaluationResult quasiResult = runQuasiClique(g);

                    // Print results
                    printResult("L-RMC", clusterSize, pIn, pOut, lrmcResult);
                    printResult("k-core", clusterSize, pIn, pOut, kcoreResult);
                    printResult("Densest", clusterSize, pIn, pOut, densestResult);
                    // printResult("Quasi-clique", clusterSize, pIn, pOut, quasiResult);
                }
            }
        }
    }

    static void printResult(String method, int clusterSize, double pIn, double pOut, EvaluationResult result) {
        System.out.printf("%s,%d,%.1f,%.2f,%.3f,%.3f,%.3f,%.3f,%d,%d%n",
            method, clusterSize, pIn, pOut,
            result.precision, result.recall, result.f1,
            result.density, result.rmcScore, result.runtimeMs);
    }
}
