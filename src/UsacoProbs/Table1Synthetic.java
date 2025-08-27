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

    static class AggregatedResult {
        double precision, recall, f1;
        double minDeg, avgDeg, density;
        long rmcScore;
        long runtimeMs;
        int count;

        AggregatedResult() {
            precision = recall = f1 = 0;
            minDeg = avgDeg = density = 0;
            rmcScore = 0;
            runtimeMs = 0;
            count = 0;
        }

        void addResult(EvaluationResult r) {
            precision += r.precision;
            recall += r.recall;
            f1 += r.f1;
            minDeg += r.minDeg;
            avgDeg += r.avgDeg;
            density += r.density;
            rmcScore += r.found.size() * r.minDeg;
            runtimeMs += r.runtimeMs;
            count++;
        }

        void average() {
            if (count > 0) {
                precision /= count;
                recall /= count;
                f1 /= count;
                minDeg /= count;
                avgDeg /= count;
                density /= count;
                rmcScore /= count;
                runtimeMs /= count;
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

    /**
     * Generates a simple synthetic graph with a single cluster for testing purposes.
     *
     * @param nTotal The total number of nodes in the graph.
     * @param clusterSize The size of the single cluster.
     * @param pIn The probability of edges within the cluster.
     * @param pOut The probability of edges from the cluster to the rest of the graph.
     * @param rand The random number generator used for probabilistic edge creation.
     * @return A synthetic graph with a single cluster.
     */
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
            clique2_mk_benchmark_accuracy.runLaplacianRMC(g.adj, eps);

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

    // Sweep parameters and write CSV like the reference file
    public static void main(String[] args) {
        double eps = 10; // Default epsilon value
        int numRuns = 10; // Default number of runs

        if (args.length > 0) {
            try {
                eps = Double.parseDouble(args[0]);
            } catch (NumberFormatException e) {
                System.err.println("Invalid epsilon value. Using default value of 10.");
            }
        }

        try {
            runAndWriteCsv(eps, numRuns);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static void runAndWriteCsv(double eps, int numRuns) throws IOException {
        final int nTotal = 2500;

        // Parameter sweep
        int[] clusterSizes = {100, 150, 200};
        double[] pIns = {0.6, 0.7, 0.8, 0.9};
        double[] pOuts = {0.25, 0.35, 0.45};

        String outPath = String.format(
                java.util.Locale.US,
                "UsacoProbs/Hard-nTotal=%d,eps=%.1e.csv",
                nTotal, eps);

        try (java.io.PrintWriter w = new java.io.PrintWriter(new java.io.BufferedWriter(new java.io.FileWriter(outPath)))) {
            w.println("Method,ClusterSize,InternalDensity,ExternalDensity,Precision,Recall,F1,Density,RMCScore,Runtime");

            for (int k : clusterSizes) {
                for (double pIn : pIns) {
                    for (double pOut : pOuts) {
                        // Run multiple times with different seeds
                        AggregatedResult lrmcAgg = new AggregatedResult();
                        AggregatedResult kcoreAgg = new AggregatedResult();
                        AggregatedResult densestAgg = new AggregatedResult();

                        for (int run = 0; run < numRuns; run++) {
                            int seed = 42 + run;
                            Random rand = new Random(seed);
                            SyntheticGraph g = generatePlantedClusterGraph(nTotal, k, pIn, pOut, rand);

                            EvaluationResult lrmc = runSingleLRMC(g, eps);
                            EvaluationResult kcore = runBestKCore(g);
                            EvaluationResult densest = runDensestSingle(g);

                            lrmcAgg.addResult(lrmc);
                            kcoreAgg.addResult(kcore);
                            densestAgg.addResult(densest);
                        }

                        // Average the results
                        lrmcAgg.average();
                        kcoreAgg.average();
                        densestAgg.average();

                        // Write averaged results
                        writeAggregatedRow(w, "L-RMC", k, pIn, pOut, lrmcAgg);
                        writeAggregatedRow(w, "k-core", k, pIn, pOut, kcoreAgg);
                        writeAggregatedRow(w, "Densest", k, pIn, pOut, densestAgg);
                        System.out.printf("Finished instance: k=%d, pIn=%.1f, pOut=%.2f (averaged over %d runs)%n", k, pIn, pOut, numRuns);
                        System.out.flush();
                    }
                }
            }
        }
    }

    private static void writeAggregatedRow(java.io.PrintWriter w, String method, int k, double pIn, double pOut, AggregatedResult r) {
        w.printf(java.util.Locale.US,
                "%s,%d,%.1f,%.2f,%.3f,%.3f,%.3f,%.3f,%d,%d%n",
                method, k, pIn, pOut, r.precision, r.recall, r.f1, r.density, r.rmcScore, r.runtimeMs);
    }

    private static void writeRow(java.io.PrintWriter w, String method, int k, double pIn, double pOut, EvaluationResult r) {
        long rmcScore = (long) (r.found.size() * r.minDeg);
        w.printf(java.util.Locale.US,
                "%s,%d,%.1f,%.2f,%.3f,%.3f,%.3f,%.3f,%d,%d%n",
                method, k, pIn, pOut, r.precision, r.recall, r.f1, r.density, rmcScore, r.runtimeMs);
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

    // New: planted single cluster with background noise, parameterized by pIn and pBackground
    static SyntheticGraph generatePlantedClusterGraph(int nTotal, int clusterSize, double pIn, double pBackground, Random rand) {
        SyntheticGraph g = new SyntheticGraph(nTotal);
        for (int i = 1; i <= clusterSize; i++) g.plantedCluster.add(i);

        // Intra-cluster edges with probability pIn
        for (int i = 1; i <= clusterSize; i++) {
            for (int j = i + 1; j <= clusterSize; j++) {
                if (rand.nextDouble() < pIn) g.addEdge(i, j);
            }
        }

        // Cluster-to-background edges with probability pBackground
        for (int i = 1; i <= clusterSize; i++) {
            for (int j = clusterSize + 1; j <= nTotal; j++) {
                if (rand.nextDouble() < pBackground) g.addEdge(i, j);
            }
        }

        // Background-background edges with probability pBackground
        for (int i = clusterSize + 1; i <= nTotal; i++) {
            for (int j = i + 1; j <= nTotal; j++) {
                if (rand.nextDouble() < pBackground) g.addEdge(i, j);
            }
        }
        return g;
    }

    static void printResult(String scenario, String method, EvaluationResult result) {
        System.out.printf("%s,%s,%.3f,%.3f,%.3f,%d,%.0f,%.1f,%.3f,%d%n",
            scenario, method, result.precision, result.recall, result.f1,
            result.found.size(), result.minDeg, result.avgDeg,
            result.density, result.runtimeMs);
    }
}
