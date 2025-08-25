package UsacoProbs;

import java.io.*;
import java.util.*;

public class Table1Synthetic {

    static class SyntheticGraph {
        List<Integer>[] adj;
        Set<Integer> plantedCluster; // Single cluster
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
        Set<Integer> found;
        double precision, recall, f1;
        double minDeg, avgDeg, density;
        long runtimeMs;

        EvaluationResult(Set<Integer> found, Set<Integer> planted, long runtime) {
            this.found = found;
            this.runtimeMs = runtime;

            if (found.isEmpty()) {
                precision = recall = f1 = 0.0;
                minDeg = avgDeg = density = 0.0;
            } else {
                Set<Integer> intersection = new HashSet<>(found);
                intersection.retainAll(planted);

                precision = (double) intersection.size() / found.size();
                recall = planted.isEmpty() ? 0.0 : (double) intersection.size() / planted.size();
                f1 = (precision + recall == 0) ? 0.0 : 2 * precision * recall / (precision + recall);
            }
        }

        void computeStats(List<Integer>[] adj) {
            if (found.isEmpty()) {
                minDeg = avgDeg = density = 0.0;
                return;
            }

            int totalDeg = 0;
            int minD = Integer.MAX_VALUE;
            int edges = 0;

            for (int u : found) {
                int deg = 0;
                for (int v : adj[u]) {
                    if (found.contains(v)) {
                        deg++;
                        if (u < v) edges++;
                    }
                }
                totalDeg += deg;
                minD = Math.min(minD, deg);
            }

            minDeg = minD;
            avgDeg = (double) totalDeg / found.size();
            int maxEdges = found.size() * (found.size() - 1) / 2;
            density = maxEdges == 0 ? 0.0 : (double) edges / maxEdges;
        }
    }

    // Simple single-cluster graph for testing
    static SyntheticGraph generateSimpleGraph(int nTotal, int clusterSize, double pIn, double pOut, Random rand) {
        SyntheticGraph g = new SyntheticGraph(nTotal);

        // Plant one cluster in nodes 1..clusterSize
        for (int i = 1; i <= clusterSize; i++) {
            g.plantedCluster.add(i);
        }

        // Add intra-cluster edges
        for (int i = 1; i <= clusterSize; i++) {
            for (int j = i + 1; j <= clusterSize; j++) {
                if (rand.nextDouble() < pIn) {
                    g.addEdge(i, j);
                }
            }
        }

        // Add noise edges from cluster to outside
        for (int i = 1; i <= clusterSize; i++) {
            for (int j = clusterSize + 1; j <= nTotal; j++) {
                if (rand.nextDouble() < pOut) {
                    g.addEdge(i, j);
                }
            }
        }

        // Add background noise
        for (int i = clusterSize + 1; i <= nTotal; i++) {
            for (int j = i + 1; j <= nTotal; j++) {
                if (rand.nextDouble() < 0.01) {
                    g.addEdge(i, j);
                }
            }
        }

        return g;
    }

    static EvaluationResult runSingleLRMC(SyntheticGraph g, double eps) {
        long start = System.nanoTime();

        clique2_mk_benchmark_accuracy.Result result =
            clique2_mk_benchmark_accuracy.runLaplacianRMC(g.adj, g.n, eps);

        long end = System.nanoTime();

        Set<Integer> found = (result.bestComponent != null) ? result.bestComponent : new HashSet<>();
        EvaluationResult evalResult = new EvaluationResult(found, g.plantedCluster, (end - start) / 1_000_000);
        evalResult.computeStats(g.adj);
        return evalResult;
    }

    static EvaluationResult runBestKCore(SyntheticGraph g) {
        long start = System.nanoTime();

        Set<Integer> bestCore = new HashSet<>();
        double bestF1 = 0.0;

        // Try different k values
        for (int k = 30; k >= 3; k--) {
            Set<Integer> core = computeKCore(g, k);
            if (core.isEmpty()) continue;

            // Evaluate this core
            Set<Integer> intersection = new HashSet<>(core);
            intersection.retainAll(g.plantedCluster);

            double precision = core.isEmpty() ? 0 : (double) intersection.size() / core.size();
            double recall = g.plantedCluster.isEmpty() ? 0 : (double) intersection.size() / g.plantedCluster.size();
            double f1 = (precision + recall == 0) ? 0 : 2 * precision * recall / (precision + recall);

            if (f1 > bestF1) {
                bestF1 = f1;
                bestCore = new HashSet<>(core);
            }
        }

        long end = System.nanoTime();
        EvaluationResult evalResult = new EvaluationResult(bestCore, g.plantedCluster, (end - start) / 1_000_000);
        evalResult.computeStats(g.adj);
        return evalResult;
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

    static EvaluationResult runDensestSingle(SyntheticGraph g) {
        long start = System.nanoTime();

        Set<Integer> remaining = new HashSet<>();
        for (int i = 1; i <= g.n; i++) {
            remaining.add(i);
        }

        Set<Integer> bestSubgraph = new HashSet<>();
        double bestF1 = 0.0;

        while (remaining.size() > 10) {
            // Evaluate current subgraph
            Set<Integer> intersection = new HashSet<>(remaining);
            intersection.retainAll(g.plantedCluster);

            double precision = remaining.isEmpty() ? 0 : (double) intersection.size() / remaining.size();
            double recall = g.plantedCluster.isEmpty() ? 0 : (double) intersection.size() / g.plantedCluster.size();
            double f1 = (precision + recall == 0) ? 0 : 2 * precision * recall / (precision + recall);

            if (f1 > bestF1) {
                bestF1 = f1;
                bestSubgraph = new HashSet<>(remaining);
            }

            // Remove min degree node
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
        EvaluationResult evalResult = new EvaluationResult(bestSubgraph, g.plantedCluster, (end - start) / 1_000_000);
        evalResult.computeStats(g.adj);
        return evalResult;
    }

    // Scenario 1: Hub-dominated graph
    static SyntheticGraph generateHubGraph(int nTotal, int clusterSize, double pIn, Random rand) {
        SyntheticGraph g = new SyntheticGraph(nTotal);

        // Plant target cluster (nodes 1..clusterSize)
        for (int i = 1; i <= clusterSize; i++) {
            g.plantedCluster.add(i);
        }

        // Make cluster internally connected but with uneven degrees
        // Core nodes (first half) are very well connected
        int coreSize = clusterSize / 2;
        for (int i = 1; i <= coreSize; i++) {
            for (int j = i + 1; j <= coreSize; j++) {
                if (rand.nextDouble() < pIn) {
                    g.addEdge(i, j);
                }
            }
        }

        // Peripheral nodes connect to core but not each other much
        for (int i = coreSize + 1; i <= clusterSize; i++) {
            // Each peripheral node connects to 4-6 core nodes
            int connections = 4 + rand.nextInt(3);
            Set<Integer> connected = new HashSet<>();
            while (connected.size() < connections && connected.size() < coreSize) {
                int target = 1 + rand.nextInt(coreSize);
                if (!connected.contains(target)) {
                    g.addEdge(i, target);
                    connected.add(target);
                }
            }

            // Very few connections to other peripheral nodes
            for (int j = i + 1; j <= clusterSize; j++) {
                if (rand.nextDouble() < 0.1) {
                    g.addEdge(i, j);
                }
            }
        }

        // Add hub nodes that connect to MANY nodes (this confuses density-based methods)
        for (int h = 0; h < 3; h++) {
            int hub = nTotal - h;

            // Hub connects to 60% of target cluster
            for (int i = 1; i <= clusterSize; i++) {
                if (rand.nextDouble() < 0.6) {
                    g.addEdge(hub, i);
                }
            }

            // Hub also connects to many random nodes
            for (int i = clusterSize + 1; i < nTotal - 3; i++) {
                if (rand.nextDouble() < 0.3) {
                    g.addEdge(hub, i);
                }
            }
        }

        // Add background noise
        for (int i = clusterSize + 1; i < nTotal - 3; i++) {
            for (int j = i + 1; j < nTotal - 3; j++) {
                if (rand.nextDouble() < 0.02) {
                    g.addEdge(i, j);
                }
            }
        }

        return g;
    }

    // Scenario 2: Competing clusters of different types
    static SyntheticGraph generateCompetingClusters(int nTotal, int targetSize, Random rand) {
        SyntheticGraph g = new SyntheticGraph(nTotal);

        // Target cluster: high minimum degree but moderate average degree
        for (int i = 1; i <= targetSize; i++) {
            g.plantedCluster.add(i);
        }

        // Make target cluster with good minimum degree
        for (int i = 1; i <= targetSize; i++) {
            // Each node connects to exactly 8-12 others (good min degree)
            Set<Integer> neighbors = new HashSet<>();
            while (neighbors.size() < 8 + rand.nextInt(5)) {
                int j = 1 + rand.nextInt(targetSize);
                if (j != i && !neighbors.contains(j)) {
                    neighbors.add(j);
                    g.addEdge(i, j);
                }
            }
        }

        // Competing cluster: high average degree but poor minimum degree (has weak nodes)
        int compStart = targetSize + 10;
        int compSize = targetSize + 5; // Slightly larger

        // Most nodes in competing cluster are very well connected
        for (int i = compStart; i < compStart + compSize - 5; i++) {
            for (int j = i + 1; j < compStart + compSize - 5; j++) {
                if (rand.nextDouble() < 0.8) { // Very dense core
                    g.addEdge(i, j);
                }
            }
        }

        // But a few nodes have very low degree (this hurts RMC score but not density)
        for (int i = compStart + compSize - 5; i < compStart + compSize; i++) {
            // These nodes connect to only 2-3 others
            int connections = 2 + rand.nextInt(2);
            for (int c = 0; c < connections; c++) {
                int target = compStart + rand.nextInt(compSize - 5);
                g.addEdge(i, target);
            }
        }

        // Add inter-cluster noise
        for (int i = 1; i <= targetSize; i++) {
            if (rand.nextDouble() < 0.05) {
                int target = compStart + rand.nextInt(compSize);
                g.addEdge(i, target);
            }
        }

        return g;
    }

    // Add this debugging method
    static void debugClusters(SyntheticGraph g, String scenario) {
        System.err.printf("\n=== DEBUG %s ===\n", scenario);

        // Find all connected components of reasonable size
        boolean[] visited = new boolean[g.n + 1];
        List<Set<Integer>> components = new ArrayList<>();

        for (int i = 1; i <= g.n; i++) {
            if (!visited[i]) {
                Set<Integer> component = new HashSet<>();
                Queue<Integer> queue = new LinkedList<>();
                queue.offer(i);
                visited[i] = true;

                while (!queue.isEmpty()) {
                    int u = queue.poll();
                    component.add(u);

                    for (int v : g.adj[u]) {
                        if (!visited[v]) {
                            visited[v] = true;
                            queue.offer(v);
                        }
                    }
                }

                if (component.size() >= 10) { // Only consider large components
                    components.add(component);
                }
            }
        }

        System.err.printf("Found %d large components\n", components.size());

        // Analyze each component
        for (int i = 0; i < components.size(); i++) {
            Set<Integer> comp = components.get(i);

            // Compute statistics
            int edges = 0;
            int minDeg = Integer.MAX_VALUE;
            int totalDeg = 0;

            for (int u : comp) {
                int deg = 0;
                for (int v : g.adj[u]) {
                    if (comp.contains(v)) {
                        deg++;
                        if (u < v) edges++;
                    }
                }
                totalDeg += deg;
                minDeg = Math.min(minDeg, deg);
            }

            double avgDeg = (double) totalDeg / comp.size();
            double density = comp.size() <= 1 ? 0 : (double) edges / (comp.size() * (comp.size() - 1) / 2);
            int rmcScore = comp.size() * minDeg;

            // Check overlap with planted cluster
            Set<Integer> intersection = new HashSet<>(comp);
            intersection.retainAll(g.plantedCluster);
            double precision = intersection.size() / (double) comp.size();
            double recall = intersection.size() / (double) g.plantedCluster.size();

            System.err.printf("Component %d: size=%d, minDeg=%d, avgDeg=%.1f, density=%.3f, RMC=%d, precision=%.3f, recall=%.3f\n",
                i, comp.size(), minDeg, avgDeg, density, rmcScore, precision, recall);

            // Show some node IDs to see which component is which
            List<Integer> nodeList = new ArrayList<>(comp);
            nodeList.sort(Integer::compareTo);
            System.err.printf("  First 10 nodes: %s\n", nodeList.subList(0, Math.min(10, nodeList.size())));
        }

        // Planted cluster stats
        System.err.printf("PLANTED cluster: size=%d, first 10 nodes: %s\n",
            g.plantedCluster.size(),
            g.plantedCluster.stream().sorted().limit(10).collect(java.util.stream.Collectors.toList()));
    }

    // Also test epsilon values systematically
    static void testEpsilonValues(SyntheticGraph g, String scenario) {
        System.err.printf("\n=== EPSILON TEST %s ===\n", scenario);

        double[] epsilons = {10, 50, 100, 500, 1000, 2000, 5000};

        for (double eps : epsilons) {
            clique2_mk_benchmark_accuracy.Result result =
                clique2_mk_benchmark_accuracy.runLaplacianRMC(g.adj, g.n, eps);

            if (result.bestComponent != null && !result.bestComponent.isEmpty()) {
                Set<Integer> found = result.bestComponent;
                Set<Integer> intersection = new HashSet<>(found);
                intersection.retainAll(g.plantedCluster);

                double precision = intersection.size() / (double) found.size();
                double recall = intersection.size() / (double) g.plantedCluster.size();

                // Show first few nodes to see which cluster was found
                List<Integer> nodeList = new ArrayList<>(found);
                nodeList.sort(Integer::compareTo);

                System.err.printf("eps=%.0f: size=%d, precision=%.3f, recall=%.3f, score=%.3f, first5=%s\n",
                    eps, found.size(), precision, recall, result.bestScore,
                    nodeList.subList(0, Math.min(5, nodeList.size())));
            } else {
                System.err.printf("eps=%.0f: NOTHING FOUND\n", eps);
            }
        }
    }

    // Add this method to debug L-RMC specifically
    static void debugLRMCSteps(SyntheticGraph g, String scenario) {
        System.err.printf("\n=== L-RMC STEP DEBUG %s ===\n", scenario);

        // Test different epsilon values specifically for this graph
        double[] epsilons = {50, 100, 200, 500, 1000, 2000, 5000, 10000};

        for (double eps : epsilons) {
            clique2_mk_benchmark_accuracy.Result result =
                clique2_mk_benchmark_accuracy.runLaplacianRMC(g.adj, g.n, eps);

            if (result.bestComponent != null && !result.bestComponent.isEmpty()) {
                Set<Integer> found = result.bestComponent;

                // Compute RMC score manually
                int minDeg = Integer.MAX_VALUE;
                for (int u : found) {
                    int deg = 0;
                    for (int v : g.adj[u]) {
                        if (found.contains(v)) deg++;
                    }
                    minDeg = Math.min(minDeg, deg);
                }
                int rmcScore = found.size() * minDeg;

                // Check which component this overlaps with most
                String whichComponent = "Unknown";
                if (found.stream().anyMatch(x -> x <= 30)) {
                    whichComponent = "Target";
                } else if (found.stream().anyMatch(x -> x >= 100)) {
                    whichComponent = "Competing";
                }

                System.err.printf("eps=%.0f: size=%d, minDeg=%d, RMC=%d, score=%.6f, type=%s\n",
                    eps, found.size(), minDeg, rmcScore, result.bestScore, whichComponent);
            } else {
                System.err.printf("eps=%.0f: NOTHING FOUND\n", eps);
            }
        }
    }

    // Modified main to include this debug
    public static void main(String[] args) {
        Random rand = new Random(42);

        System.out.println("Scenario,Method,Precision,Recall,F1,FoundSize,MinDeg,AvgDeg,Density,Runtime");

        // Only test the broken case
        SyntheticGraph twoClusters = generateTwoSeparateClusters(rand);

        debugClusters(twoClusters, "TwoClusters");
        debugLRMCSteps(twoClusters, "TwoClusters");

        testAllMethods("TwoClusters", twoClusters, 500);
    }

    // Scenario 1: Star graph vs clique - L-RMC should prefer the clique
    static SyntheticGraph generateStarVsClique(Random rand) {
        SyntheticGraph g = new SyntheticGraph(100);

        // Target: A 20-clique (nodes 1-20)
        for (int i = 1; i <= 20; i++) {
            g.plantedCluster.add(i);
            for (int j = i + 1; j <= 20; j++) {
                g.addEdge(i, j);
            }
        }

        // Distractor: A big star (node 50 connected to nodes 51-80)
        for (int i = 51; i <= 80; i++) {
            g.addEdge(50, i);
        }

        // Note: These are SEPARATE - no edges between them
        return g;
    }

    // Scenario 2: Hub connected to clique
    static SyntheticGraph generateHubClique(Random rand) {
        SyntheticGraph g = new SyntheticGraph(100);

        // Target: Dense cluster (nodes 1-25)
        for (int i = 1; i <= 25; i++) {
            g.plantedCluster.add(i);
            for (int j = i + 1; j <= 25; j++) {
                if (rand.nextDouble() < 0.7) { // 70% density
                    g.addEdge(i, j);
                }
            }
        }

        // Hub that connects to the cluster (this inflates average degree)
        for (int i = 1; i <= 25; i++) {
            if (rand.nextDouble() < 0.8) { // Hub connects to 80% of cluster
                g.addEdge(99, i);
            }
        }

        return g;
    }

    // Scenario 3: Two completely separate clusters
    static SyntheticGraph generateTwoSeparateClusters(Random rand) {
        SyntheticGraph g = new SyntheticGraph(200);

        // Target cluster: Good minimum degree (nodes 1-30)
        for (int i = 1; i <= 30; i++) {
            g.plantedCluster.add(i);
        }

        // Each node connects to exactly 12 others (good minimum degree)
        for (int i = 1; i <= 30; i++) {
            Set<Integer> neighbors = new HashSet<>();
            while (neighbors.size() < 12) {
                int j = 1 + rand.nextInt(30);
                if (j != i) neighbors.add(j);
            }
            for (int j : neighbors) {
                g.addEdge(i, j);
            }
        }

        // Competing cluster: High average degree but poor minimum degree (nodes 100-140)
        // Core is very dense
        for (int i = 100; i <= 130; i++) {
            for (int j = i + 1; j <= 130; j++) {
                if (rand.nextDouble() < 0.8) {
                    g.addEdge(i, j);
                }
            }
        }
        // But a few nodes have very low degree
        for (int i = 131; i <= 140; i++) {
            // These connect to only 2 nodes
            g.addEdge(i, 100 + rand.nextInt(5));
            g.addEdge(i, 100 + rand.nextInt(5));
        }

        return g;
    }

    static void testAllMethods(String scenario, SyntheticGraph g, double eps) {
        // Debug what we created
        debugClusters(g, scenario);

        EvaluationResult lrmcResult = runSingleLRMC(g, eps);
        EvaluationResult kcoreResult = runBestKCore(g);
        EvaluationResult densestResult = runDensestSingle(g);

        printResult(scenario, "L-RMC", lrmcResult);
        printResult(scenario, "k-core", kcoreResult);
        printResult(scenario, "Densest", densestResult);
    }

    static void printResult(String scenario, String method, EvaluationResult result) {
        System.out.printf("%s,%s,%.3f,%.3f,%.3f,%d,%.0f,%.1f,%.3f,%d%n",
            scenario, method, result.precision, result.recall, result.f1,
            result.found.size(), result.minDeg, result.avgDeg,
            result.density, result.runtimeMs);
    }
}
