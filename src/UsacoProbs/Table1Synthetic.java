package UsacoProbs;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

/**
 * Table1Synthetic
 *
 * Reproducible harness for "Clustering on synthetic graphs" (Table 1).
 * Generates planted-cluster graphs, runs baselines (k-core, densest-subgraph, quasi-clique),
 * optionally runs L-RMC (clique2_mk) if configured, and computes NMI, ARI, and F1 (Hungarian matched).
 *
 * Output: CSV with one row per trial x method: method,n,k,p_intra,p_inter,seed,NMI,ARI,F1
 *
 * Notes:
 *  - For L-RMC integration, see the "L-RMC integration" section below.
 *  - Node IDs are 0..n-1 internally. When writing edge lists, 1-based IDs are used
 *    to match your existing clique2_mk/LRMCmkpaper format.
 */
public class Table1Synthetic {

    // ------------------------------ Experiment config ------------------------------
    static final int[] N_LIST = new int[]{100_000};                    // Use 1e5 for Table 1
    static final int NUM_CLUSTERS = 10;
    static final double CLUSTER_FRACTION = 0.20;                       // 20% of nodes belong to planted clusters

    // --- Multiple Settings ---

    // static final double[] PINTRA_LIST = new double[]{0.005, 0.010, 0.020};
    // static final double[] PINTER_LIST = new double[]{1e-4, 3e-4, 1e-3};
    // static final int TRIALS_PER_SETTING = 3;                           // Increase to 5–10 once stable

    // ---- Single Setting ----

    static final double[] PINTRA_LIST = new double[]{0.020};
    static final double[] PINTER_LIST = new double[]{1.5e-4};
    static final int TRIALS_PER_SETTING = 5;

    // ------------------------

    static final long BASE_SEED = 42L;

    // Methods to run (toggle on/off)
    static final boolean RUN_KCORE = true;
    static final boolean RUN_DENSEST = true;
    static final boolean RUN_QUASICLIQUE = true;
    static final boolean RUN_LRMC = true;

    // L-RMC integration: update these names to match your build
    static String LRMC_MAIN_CLASS = "UsacoProbs.clique2_mk_benchmark_accuracy";
    static final String EXTRA_HEAP = "-Xmx16g";                        // adjust if needed
    static final double LRMC_EPSILON = 1e-6;                           // same epsilon used in the paper

    // Output
    static final Path OUT_CSV = Paths.get("table1_synthetic.csv");
    static final Path TMP_DIR = Paths.get("tmp_table1");

    // ------------------------------- Data model -------------------------------
    static final class Graph {
        final int n;
        final int m;
        final int[][] adj;
        final int[] truth;

        Graph(int n, int[][] adj, int[] truth, int m) {
            this.n = n;
            this.adj = adj;
            this.truth = truth;
            this.m = m;
        }
    }

    // ------------------------------- Main entry -------------------------------
    public static void main(String[] args) throws Exception {
        // Parse command line arguments
        if (args.length > 0) {
            if (args[0].equals("-h") || args[0].equals("--help")) {
                System.out.println("Usage: java UsacoProbs.Table1Synthetic [LRMC_MAIN_CLASS]");
                System.out.println("  LRMC_MAIN_CLASS: Full class path for L-RMC implementation (default: UsacoProbs.clique2_mk_benchmark_accuracy)");
                System.out.println("Example: java UsacoProbs.Table1Synthetic UsacoProbs.clique2_mk_benchmark_accuracy");
                return;
            }
            LRMC_MAIN_CLASS = args[0];
            System.out.println("Using LRMC main class: " + LRMC_MAIN_CLASS);
        }
        Files.createDirectories(TMP_DIR);

        // Total trials (for progress tracking)
        int totalTrials = N_LIST.length * PINTRA_LIST.length * PINTER_LIST.length * TRIALS_PER_SETTING;
        int completedTrials = 0;

        try (BufferedWriter out = Files.newBufferedWriter(OUT_CSV, StandardCharsets.UTF_8)) {
            out.write("method,n,k,p_intra,p_inter,seed,NMI,ARI,F1\n");

            for (int n : N_LIST) {
                for (double pIntra : PINTRA_LIST) {
                    for (double pInter : PINTER_LIST) {
                        for (int tr = 0; tr < TRIALS_PER_SETTING; tr++) {
                            long seed = BASE_SEED + tr;
                            Graph G = generatePlantedClusters(n, NUM_CLUSTERS, CLUSTER_FRACTION, pIntra, pInter, seed);

                            if (RUN_KCORE) {
                                int[][] clusters = kcoreTopK(G, NUM_CLUSTERS);
                                double[] s = evalAll(G.truth, clusters, G.n);
                                writeRow(out, "k-core", n, NUM_CLUSTERS, pIntra, pInter, seed, s);
                            }
                            if (RUN_DENSEST) {
                                int[][] clusters = densestTopK(G, NUM_CLUSTERS);
                                double[] s = evalAll(G.truth, clusters, G.n);
                                writeRow(out, "densest", n, NUM_CLUSTERS, pIntra, pInter, seed, s);
                            }
                            if (RUN_QUASICLIQUE) {
                                int[][] clusters = quasiCliqueTopK(G, NUM_CLUSTERS);
                                double[] s = evalAll(G.truth, clusters, G.n);
                                writeRow(out, "quasi-clique", n, NUM_CLUSTERS, pIntra, pInter, seed, s);
                            }
                            if (RUN_LRMC) {
                                // L-RMC via external process (clique2_mk) after adding membership output support.
                                int[][] clusters = lrmcTopK_viaCli(G, NUM_CLUSTERS, seed);
                                double[] s = evalAll(G.truth, clusters, G.n);
                                writeRow(out, "L-RMC", n, NUM_CLUSTERS, pIntra, pInter, seed, s);
                            }

                            // Progress tracking
                            completedTrials++;
                            if (completedTrials % 5 == 0 || completedTrials == totalTrials) {
                                System.out.printf("Progress: %d/%d trials completed (%.1f%%)%n",
                                    completedTrials, totalTrials, (100.0 * completedTrials) / totalTrials);
                            }
                        }
                    }
                }
            }
        }

        System.out.println("Wrote " + OUT_CSV.toAbsolutePath());
        System.out.println("Next: aggregate means/std and render the LaTeX row for Table 1.");
    }

    private static void writeRow(BufferedWriter out, String method, int n, int k, double pIntra, double pInter, long seed, double[] s) throws IOException {
        out.write(method + "," + n + "," + k + "," + pIntra + "," + pInter + "," + seed + "," +
                fmt(s[0]) + "," + fmt(s[1]) + "," + fmt(s[2]) + "\n");
        out.flush();
    }
    private static String fmt(double x) { return String.format(Locale.US, "%.6f", x); }

    // --------------------------- Synthetic generator ---------------------------
    /**
     * Planted K clusters in an ER(n, q) background.
     *  - Choose K cluster sizes that sum to ~n * CLUSTER_FRACTION with ±20% jitter.
     *  - Within-cluster edges appear with pIntra.
     *  - Between any two nodes not in the same cluster, edges appear with pInter.
     * Returns adjacency and an int[] truth with labels in 0..K (0 = background).
     */

    static Graph generatePlantedClusters(int n, int K, double frac, double pIntra, double pInter, long seed) throws IOException {
        Random rng = new Random(seed);

        int totalClusterNodes = (int) Math.round(n * frac);
        int[] sizes = new int[K];
        int base = Math.max(5, totalClusterNodes / K);
        int rem = totalClusterNodes - base * K;
        for (int i = 0; i < K; i++) sizes[i] = base + (i < rem ? 1 : 0);
        // ±20% jitter, preserving sum approximately
        int targeted = 0;
        for (int i = 0; i < K; i++) {
            int jitter = (int) Math.round((rng.nextDouble() * 0.4 - 0.2) * sizes[i]);
            sizes[i] = Math.max(5, sizes[i] + jitter);
            targeted += sizes[i];
        }
        // normalize total to <= totalClusterNodes
        int diff = targeted - totalClusterNodes;
        for (int i = 0; diff > 0 && i < K; i++, diff--) sizes[i] = Math.max(5, sizes[i] - 1);

        // Assign labels and record cluster node lists
        int[] truth = new int[n];
        Arrays.fill(truth, 0); // background
        int[][] clusterNodes = new int[K][];
        int next = 0;
        for (int c = 0; c < K; c++) {
            int sz = Math.min(sizes[c], n - next);
            clusterNodes[c] = new int[sz];
            for (int t = 0; t < sz; t++) {
                clusterNodes[c][t] = next;
                truth[next] = c + 1;
                next++;
            }
        }
        int bgStart = next; // nodes [bgStart..n-1] are background

        // Sample edges with geometric skipping, accumulate in edge lists
        IntList EU = new IntList();
        IntList EV = new IntList();

        // within clusters
        for (int c = 0; c < K; c++) {
            int[] S = clusterNodes[c];
            sampleTriPairs(S, pIntra, rng, EU, EV);
        }
        // cluster-to-cluster
        for (int c1 = 0; c1 < K; c1++) {
            for (int c2 = c1 + 1; c2 < K; c2++) {
                sampleRectPairs(clusterNodes[c1], clusterNodes[c2], pInter, rng, EU, EV);
            }
        }
        // cluster-to-background
        int[] BG = new int[n - bgStart];
        for (int i = 0; i < BG.length; i++) BG[i] = bgStart + i;
        for (int c = 0; c < K; c++) sampleRectPairs(clusterNodes[c], BG, pInter, rng, EU, EV);
        // background-to-background
        sampleTriPairs(BG, pInter, rng, EU, EV);

        // Build adjacency from edge lists
        int[] deg = new int[n];
        for (int i = 0; i < EU.size(); i++) { deg[EU.get(i)]++; deg[EV.get(i)]++; }
        int[][] adj = new int[n][];
        int[] ptr = new int[n];
        for (int i = 0; i < n; i++) adj[i] = new int[deg[i]];
        for (int i = 0; i < EU.size(); i++) {
            int u = EU.get(i), v = EV.get(i);
            adj[u][ptr[u]++] = v;
            adj[v][ptr[v]++] = u;
        }
        long m = EU.size(); // undirected edge count

        return new Graph(n, adj, truth, (int) m);
    }

    // Sample edges among pairs i<j from a single set S using geometric skipping
    static void sampleTriPairs(int[] S, double p, Random rng, IntList EU, IntList EV) {
        int s = S.length;
        long T = (long) s * (s - 1) / 2L; // number of pairs
        if (T <= 0 || p <= 0) return;
        double log1mp = Math.log1p(-p);
        long t = nextGeom(rng, log1mp);
        while (t < T) {
            int[] ij = triIndexToPair(t, s);
            int i = S[ij[0]];
            int j = S[ij[1]];
            EU.add(i); EV.add(j);
            long jump = 1 + nextGeom(rng, log1mp);
            t += jump;
        }
    }

    // Sample edges across a rectangle A x B using geometric skipping
    static void sampleRectPairs(int[] A, int[] B, double p, Random rng, IntList EU, IntList EV) {
        int a = A.length, b = B.length;
        long T = (long) a * b;
        if (T <= 0 || p <= 0) return;
        double log1mp = Math.log1p(-p);
        long t = nextGeom(rng, log1mp);
        while (t < T) {
            int i = (int) (t / b);
            int j = (int) (t % b);
            EU.add(A[i]); EV.add(B[j]);
            long jump = 1 + nextGeom(rng, log1mp);
            t += jump;
        }
    }

    // Geometric skip: number of failures before next success for Bernoulli(p)
    static long nextGeom(Random rng, double log1mp) {
        // For u ~ U(0,1), floor(log(u)/log(1-p)) is geometric with support {0,1,2,...}
        double u = rng.nextDouble();
        if (u == 0) return Long.MAX_VALUE / 4; // extremely unlikely
        return (long) Math.floor(Math.log(u) / log1mp);
    }

    // Map t in [0, s*(s-1)/2) to pair (i,j) with 0<=i<j<s using triangular numbers
    static int[] triIndexToPair(long t, int s) {
        // Find i s.t. S(i) <= t < S(i+1), where S(i) = i*(2s - i - 1)/2
        // Solve quadratic for i
        double sd = s;
        double D = (2*sd - 1)*(2*sd - 1) - 8.0*t;
        int i = (int) Math.floor((2*sd - 1 - Math.sqrt(Math.max(0, D))) / 2.0);
        long Si = (long) i * (2L*s - i - 1L) / 2L;
        long offset = t - Si;
        int j = i + 1 + (int) offset;
        return new int[]{i, j};
    }

    // ------------------------------ Baselines ------------------------------
    // Helper: remove a set of nodes and return induced subgraph mapping old->new and new->old
    static class SubgraphView {
        final int[][] adj;
        final int[] oldId;      // new -> old
        final int[] map;        // old -> new, -1 if removed
        SubgraphView(int[][] adj, int[] oldId, int[] map) { this.adj = adj; this.oldId = oldId; this.map = map; }
    }
    static SubgraphView inducedSubgraph(int[][] adj, boolean[] removed) {
        int n = adj.length;
        int[] map = new int[n];
        Arrays.fill(map, -1);
        int cnt = 0;
        for (int i = 0; i < n; i++) if (!removed[i]) map[i] = cnt++;
        int[][] adj2 = new int[cnt][];
        for (int i = 0; i < n; i++) if (!removed[i]) {
            int ni = map[i];
            int deg = 0;
            for (int v : adj[i]) if (!removed[v]) deg++;
            adj2[ni] = new int[deg];
        }
        int[] ptr = new int[cnt];
        for (int i = 0; i < n; i++) if (!removed[i]) {
            int ni = map[i];
            for (int v : adj[i]) if (!removed[v]) adj2[ni][ptr[ni]++] = map[v];
        }
        int[] oldId = new int[cnt];
        for (int i = 0; i < n; i++) if (!removed[i]) oldId[map[i]] = i;
        return new SubgraphView(adj2, oldId, map);
    }

    // k-core: compute core numbers, then scan descending k and take biggest components across k until K disjoint clusters.
    static int[][] kcoreTopK(Graph G, int K) {
        int n = G.n;
        int[] core = degeneracyCoreNumbers(G.adj);
        // Build unique sorted k values
        int maxCore = 0; for (int c : core) if (c > maxCore) maxCore = c;
        List<Integer> ks = new ArrayList<>();
        for (int k = maxCore; k >= 1; k--) ks.add(k);

        boolean[] used = new boolean[n];
        List<int[]> clusters = new ArrayList<>();
        for (int k : ks) {
            // BFS on nodes with core >= k, skip used
            boolean[] seen = new boolean[n];
            for (int i = 0; i < n; i++) {
                if (used[i] || seen[i] || core[i] < k) continue;
                // start a component
                IntList comp = new IntList();
                Deque<Integer> dq = new ArrayDeque<>();
                dq.add(i); seen[i] = true;
                while (!dq.isEmpty()) {
                    int u = dq.poll();
                    comp.add(u);
                    for (int v : G.adj[u]) if (!used[v] && !seen[v] && core[v] >= k) { seen[v] = true; dq.add(v); }
                }
                // keep only reasonable sizes to avoid tiny cores
                if (comp.size() >= 10) {
                    int[] cc = comp.toArray();
                    clusters.add(cc);
                    // mark used
                    for (int v : cc) used[v] = true;
                    if (clusters.size() == K) break;
                }
            }
            if (clusters.size() == K) break;
        }
        return padClusters(clusters, K);
    }

    // Densest subgraph (Charikar 2-approx): run K times, each time remove found subgraph.
    static int[][] densestTopK(Graph G, int K) {
        boolean[] removed = new boolean[G.n];
        List<int[]> list = new ArrayList<>();
        for (int it = 0; it < K; it++) {
            SubgraphView S = inducedSubgraph(G.adj, removed);
            if (S.adj.length == 0) break;
            int[] take = charikarDensest(S.adj);
            if (take.length == 0) break;
            int[] mapped = new int[take.length];
            for (int i = 0; i < take.length; i++) mapped[i] = S.oldId[take[i]];
            list.add(mapped);
            for (int v : mapped) removed[v] = true;
        }
        return padClusters(list, K);
    }

    // Quasi-clique greedy: grow from top-degree seeds; accept nodes as long as density >= gamma
    static int[][] quasiCliqueTopK(Graph G, int K) {
        // pick gamma from a small grid; in synthetic setting, 0.6 is a reasonable default
        double[] gammaGrid = new double[]{0.015, 0.02, 0.025};
        // double[] gammaGrid = new double[]{0.55, 0.60, 0.65};
        boolean[] used = new boolean[G.n];
        List<int[]> list = new ArrayList<>();
        // degree array
        int[] deg = new int[G.n]; for (int i = 0; i < G.n; i++) deg[i] = G.adj[i].length;
        Integer[] order = new Integer[G.n];
        for (int i = 0; i < G.n; i++) order[i] = i;
        Arrays.sort(order, (a,b) -> Integer.compare(deg[b], deg[a]));

        for (int o = 0; o < order.length && list.size() < K; o++) {
            int seed = order[o];
            if (used[seed]) continue;
            int[] cand = greedyQuasiCliqueFromSeed(G.adj, seed, used, gammaGrid);
            if (cand.length >= 10) {
                list.add(cand);
                for (int v : cand) used[v] = true;
            }
        }
        return padClusters(list, K);
    }

    // -------------------------- L-RMC integration hook --------------------------
    /**
     * Call your clique2_mk as an external process, assuming you add an argument to write the
     * best-cluster membership to a file named "members.txt" with lines "node_id label" where
     * label is 1 for in-cluster, 0 otherwise. Then we remove that cluster and repeat K times.
     *
     * If you haven't added that output yet, keep RUN_LRMC=false and skip this.
     */
    static int[][] lrmcTopK_viaCli(Graph G, int K, long seed) throws Exception {
        boolean[] removed = new boolean[G.n];
        List<int[]> clusters = new ArrayList<>();

        for (int it = 0; it < K; it++) {
            SubgraphView S = inducedSubgraph(G.adj, removed);
            if (S.adj.length == 0) break;

            // Write edge list in the same format as LRMCmkpaper expects: header then 1-based edges
            Path edgeFile = TMP_DIR.resolve("lrmc_graph_it" + it + ".txt");
            writeEdgeList(edgeFile, S.adj, S.oldId);

            // membership file path to be written by clique2_mk after you patch it
            Path membFile = TMP_DIR.resolve("lrmc_membership_it" + it + ".txt");

            // Build process
            List<String> cmd = new ArrayList<>();
            String javaBin = System.getProperty("java.home") + File.separator + "bin" + File.separator + "java";
            String classpath = System.getProperty("java.class.path");
            cmd.add(javaBin);
            cmd.add(EXTRA_HEAP);
            cmd.add("-cp"); cmd.add(classpath);
            cmd.add(LRMC_MAIN_CLASS);
            cmd.add(Double.toString(LRMC_EPSILON));
            cmd.add(edgeFile.toString());
            cmd.add(membFile.toString());

            ProcessBuilder pb = new ProcessBuilder(cmd);
            pb.redirectErrorStream(true);
            Process proc = pb.start();
            try (BufferedReader br = new BufferedReader(new InputStreamReader(proc.getInputStream(), StandardCharsets.UTF_8))) {
                // consume output; optional: parse runtime
                while (br.readLine() != null) { /* swallow */ }
            }
            int rc = proc.waitFor();
            if (rc != 0) throw new RuntimeException("clique2_mk exited with code " + rc);

            // Read membership, map back to original nodes, mark used and store cluster
            IntList comp = new IntList();
            try (BufferedReader br = Files.newBufferedReader(membFile, StandardCharsets.UTF_8)) {
                String s;
                while ((s = br.readLine()) != null) {
                    s = s.trim();
                    if (s.isEmpty()) continue;
                    int sp = s.indexOf(' ');
                    if (sp <= 0) continue;
                    int nid = Integer.parseInt(s.substring(0, sp));   // 0-based new id
                    int lab = Integer.parseInt(s.substring(sp + 1));  // 1 or 0
                    if (lab == 1) comp.add(S.oldId[nid]);
                }
            }
            int[] c = comp.toArray();
            if (c.length < 10) break;
            clusters.add(c);
            for (int v : c) removed[v] = true;
        }
        return padClusters(clusters, K);
    }

    static void writeEdgeList(Path file, int[][] adj, int[] newToOld) throws IOException {
        int n = adj.length;
        long m = 0;
        for (int i = 0; i < n; i++) m += adj[i].length;
        m /= 2;
        try (BufferedWriter w = Files.newBufferedWriter(file, StandardCharsets.UTF_8)) {
            w.write(n + " " + m); w.newLine();
            boolean[][] seen = null; // don't allocate; we can just write each u<v once
            for (int u = 0; u < n; u++) {
                for (int v : adj[u]) if (u < v) {
                    // convert to 1-based original IDs
                    int U = newToOld[u] + 1;
                    int V = newToOld[v] + 1;
                    w.write(U + " " + V); w.newLine();
                }
            }
        }
    }

    // ------------------------ Baseline primitives ------------------------
    // Degeneracy core numbers in O(n+m)
    static int[] degeneracyCoreNumbers(int[][] adj) {
        int n = adj.length;
        int[] deg = new int[n];
        int maxDeg = 0;
        for (int i = 0; i < n; i++) {
            deg[i] = adj[i].length;
            if (deg[i] > maxDeg) maxDeg = deg[i];
        }
        int[] bin = new int[maxDeg + 1];
        for (int d : deg) bin[d]++;
        int start = 0;
        for (int d = 0; d <= maxDeg; d++) {
            int c = bin[d];
            bin[d] = start;
            start += c;
        }
        int[] vert = new int[n];
        int[] pos = new int[n];
        for (int v = 0; v < n; v++) {
            pos[v] = bin[deg[v]];
            vert[pos[v]] = v;
            bin[deg[v]]++;
        }
        for (int d = maxDeg; d > 0; d--) bin[d] = bin[d - 1];
        bin[0] = 0;

        int[] core = Arrays.copyOf(deg, n);
        for (int i = 0; i < n; i++) {
            int v = vert[i];
            for (int u : adj[v]) if (core[u] > core[v]) {
                int du = core[u];
                int pu = pos[u];
                int pw = bin[du];
                int w = vert[pw];
                if (u != w) {
                    vert[pu] = w; pos[w] = pu;
                    vert[pw] = u; pos[u] = pw;
                }
                bin[du]++;
                core[u]--;
            }
        }
        return core;
    }

    // Charikar densest subgraph, returns node set (indices in the given subgraph)
    static int[] charikarDensest(int[][] adj) {
        int n = adj.length;
        int[] deg = new int[n];
        for (int i = 0; i < n; i++) deg[i] = adj[i].length;
        // min-heap of (deg, node). Use TreeSet keyed by (deg, node)
        TreeSet<long[]> heap = new TreeSet<>((a,b) -> {
            if (a[0] != b[0]) return Long.compare(a[0], b[0]);
            return Long.compare(a[1], b[1]);
        });
        for (int i = 0; i < n; i++) heap.add(new long[]{deg[i], i});
        boolean[] removed = new boolean[n];
        long m = 0; for (int d : deg) m += d; m /= 2;
        double bestDensity = (n == 0) ? 0 : (m / (double) n);
        boolean[] bestKeep = new boolean[n];
        Arrays.fill(bestKeep, true);

        while (!heap.isEmpty()) {
            long[] cur = heap.pollFirst();
            int v = (int) cur[1];
            if (removed[v]) continue;
            removed[v] = true;
            // update neighbors
            for (int u : adj[v]) if (!removed[u]) {
                heap.remove(new long[]{deg[u], u}); // remove old
                deg[u]--;
                heap.add(new long[]{deg[u], u});
            }
            m -= deg[v];
            n -= 1;
            if (n <= 0) break;
            double density = (n == 0) ? 0 : (m / (double) n);
            if (density > bestDensity) {
                bestDensity = density;
                Arrays.fill(bestKeep, false);
                for (int i = 0; i < removed.length; i++) if (!removed[i]) bestKeep[i] = true;
            }
        }
        IntList list = new IntList();
        for (int i = 0; i < bestKeep.length; i++) if (bestKeep[i]) list.add(i);
        return list.toArray();
    }

    static int[] greedyQuasiCliqueFromSeed(int[][] adj, int seed, boolean[] used, double[] gammaGrid) {
        int n = adj.length;
        boolean[] in = new boolean[n];
        boolean[] cand = new boolean[n];
        int[] degInS = new int[n];

        int[] bestSet = new int[]{};
        int bestSize = 0;

        for (double gamma : gammaGrid) {
            Arrays.fill(in, false);
            Arrays.fill(cand, false);
            Arrays.fill(degInS, 0);

            IntList S = new IntList();
            in[seed] = true; S.add(seed);

            for (int v : adj[seed]) if (!used[v]) { cand[v] = true; degInS[v] = 1; }
            degInS[seed] = 0;

            while (true) {
                int Ssize = S.size();
                double threshAdd = gamma * Ssize;

                // pick u with maximum marginal gain = degInS[u] - gamma*|S|
                int best = -1;
                double bestScore = 1e-12; // require strictly positive gain
                for (int u = 0; u < n; u++) if (cand[u] && !in[u]) {
                    double sc = degInS[u] - threshAdd;
                    if (sc > bestScore) { bestScore = sc; best = u; }
                }
                if (best == -1) break;

                // add best
                in[best] = true; S.add(best); cand[best] = false;
                // update degInS for neighbors
                for (int w : adj[best]) {
                    if (in[w]) { degInS[best]++; degInS[w]++; }
                    else if (!used[w]) { cand[w] = true; degInS[w]++; }
                }

                // prune nodes whose in-set degree is too low for current S
                boolean changed = true;
                while (changed) {
                    changed = false;
                    Ssize = S.size();
                    if (Ssize <= 1) break;
                    // require deg_in_S(v) >= gamma*(|S|-1)
                    double need = gamma * (Ssize - 1) - 1e-9;
                    for (int i = 0; i < S.size(); i++) {
                        int v = S.get(i);
                        if (degInS[v] < need) {
                            // remove v from S
                            in[v] = false;
                            // decrease neighbor counters
                            for (int w : adj[v]) {
                                if (in[w]) { degInS[w]--; }
                                else if (cand[w]) { degInS[w]--; }
                            }
                            // remove by swap-pop
                            S.a[i] = S.a[S.sz - 1]; S.sz--;
                            changed = true;
                            // restart scan over S after size change
                            break;
                        }
                    }
                }
            }

            if (S.size() > bestSize) {
                bestSize = S.size();
                bestSet = S.toArray();
            }
        }
        return bestSet;
    }


    static int countInternalEdges(IntList S, BitSet[] bs) {
        int e = 0;
        for (int i = 0; i < S.size(); i++) {
            int u = S.get(i);
            for (int j = i + 1; j < S.size(); j++) {
                int v = S.get(j);
                if (bs[u].get(v)) e++;
            }
        }
        return e;
    }

    // Fill missing clusters with empty entries if < K found
    static int[][] padClusters(List<int[]> list, int K) {
        int[][] out = new int[K][];
        for (int i = 0; i < K; i++) out[i] = new int[0];
        for (int i = 0; i < Math.min(K, list.size()); i++) out[i] = list.get(i);
        return out;
    }

    // ----------------------------- Metrics -----------------------------
    // Convert a list of node-index arrays to a prediction label vector 0..K
    static int[] labelsFromClusters(int n, int[][] clusters) {
        int[] pred = new int[n];
        Arrays.fill(pred, 0);
        for (int i = 0; i < clusters.length; i++) {
            int lab = i + 1;
            for (int v : clusters[i]) pred[v] = lab;
        }
        return pred;
    }

    static double[] evalAll(int[] truth, int[][] clusters, int n) {
        int[] pred = labelsFromClusters(n, clusters);
        double nmi = nmi(truth, pred);
        double ari = ari(truth, pred);
        double f1 = f1Hungarian(truth, pred);
        return new double[]{nmi, ari, f1};
    }

    // NMI with sqrt normalization: I(Y;Z)/sqrt(H(Y)H(Z))
    static double nmi(int[] y, int[] z) {
        int n = y.length;
        int Ky = 1 + max(y);
        int Kz = 1 + max(z);
        long[][] C = new long[Ky][Kz];
        long[] ry = new long[Ky];
        long[] rz = new long[Kz];
        for (int i = 0; i < n; i++) {
            C[y[i]][z[i]]++;
            ry[y[i]]++;
            rz[z[i]]++;
        }
        double I = 0;
        for (int i = 0; i < Ky; i++) {
            for (int j = 0; j < Kz; j++) {
                if (C[i][j] == 0) continue;
                double pij = C[i][j] / (double) n;
                double pi = ry[i] / (double) n;
                double pj = rz[j] / (double) n;
                I += pij * Math.log(pij / (pi * pj));
            }
        }
        double Hy = 0, Hz = 0;
        for (int i = 0; i < Ky; i++) if (ry[i] > 0) {
            double p = ry[i] / (double) n;
            Hy -= p * Math.log(p);
        }
        for (int j = 0; j < Kz; j++) if (rz[j] > 0) {
            double p = rz[j] / (double) n;
            Hz -= p * Math.log(p);
        }
        if (Hy == 0 || Hz == 0) return 0;
        double nmi = I / Math.sqrt(Hy * Hz);
        return Math.max(0, Math.min(1, nmi));
    }

    static int max(int[] a) { int m = a[0]; for (int v : a) if (v > m) m = v; return m; }

    // ARI using contingency-table formula
    static double ari(int[] y, int[] z) {
        int n = y.length;
        int Ky = 1 + max(y);
        int Kz = 1 + max(z);
        long[][] C = new long[Ky][Kz];
        long[] ai = new long[Ky];
        long[] bj = new long[Kz];
        for (int i = 0; i < n; i++) {
            C[y[i]][z[i]]++;
            ai[y[i]]++;
            bj[z[i]]++;
        }
        long sumComb = 0;
        for (int i = 0; i < Ky; i++) for (int j = 0; j < Kz; j++) sumComb += comb2(C[i][j]);
        long sumAi = 0; for (int i = 0; i < Ky; i++) sumAi += comb2(ai[i]);
        long sumBj = 0; for (int j = 0; j < Kz; j++) sumBj += comb2(bj[j]);
        long combN = comb2(n);
        double expected = (sumAi * (double) sumBj) / combN;
        double max = 0.5 * (sumAi + sumBj);
        double ari = (sumComb - expected) / (max - expected + 1e-12);
        return Math.max(-1, Math.min(1, ari));
    }
    static long comb2(long x) { return x * (x - 1) / 2; }

    // Cluster-matched F1: build bipartite weight matrix W[i][j] = F1(true_i, pred_j), then Hungarian
    static double f1Hungarian(int[] truth, int[] pred) {
        // build lists of node sets per label (ignore background=0 in matching)
        int Ky = max(truth);
        int Kz = max(pred);
        List<IntList> T = new ArrayList<>();
        List<IntList> P = new ArrayList<>();
        for (int i = 1; i <= Ky; i++) T.add(new IntList());
        for (int j = 1; j <= Kz; j++) P.add(new IntList());
        for (int v = 0; v < truth.length; v++) {
            int a = truth[v]; if (a >= 1) T.get(a - 1).add(v);
            int b = pred[v]; if (b >= 1) P.get(b - 1).add(v);
        }
        int K = Math.max(T.size(), P.size());
        double[][] W = new double[K][K];
        for (int i = 0; i < K; i++) for (int j = 0; j < K; j++) W[i][j] = 0;
        for (int i = 0; i < T.size(); i++) {
            BitSet ti = toBitset(T.get(i), truth.length);
            int si = T.get(i).size();
            for (int j = 0; j < P.size(); j++) {
                int inter = intersectCount(ti, P.get(j));
                int sj = P.get(j).size();
                double prec = sj == 0 ? 0 : inter / (double) sj;
                double rec = si == 0 ? 0 : inter / (double) si;
                double f1 = (prec + rec == 0) ? 0 : 2 * prec * rec / (prec + rec);
                W[i][j] = f1;
            }
        }
        // pad with zeros if counts differ
        double score = hungarianMax(W);
        int denom = Math.max(1, Ky); // average over true clusters
        return score / denom;
    }

    static BitSet toBitset(IntList L, int n) {
        BitSet b = new BitSet(n);
        for (int i = 0; i < L.size(); i++) b.set(L.get(i));
        return b;
    }
    static int intersectCount(BitSet a, IntList B) {
        int cnt = 0;
        for (int i = 0; i < B.size(); i++) if (a.get(B.get(i))) cnt++;
        return cnt;
    }

    // Hungarian algorithm for maximum weight assignment on a square matrix.
    static double hungarianMax(double[][] W) {
        int n = Math.max(W.length, W[0].length);
        double[][] cost = new double[n][n];
        double maxw = 0;
        for (int i = 0; i < W.length; i++) for (int j = 0; j < W[i].length; j++) if (W[i][j] > maxw) maxw = W[i][j];
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
            double w = (i < W.length && j < W[0].length) ? W[i][j] : 0;
            cost[i][j] = maxw - w;  // convert to min-cost
        }
        double[] u = new double[n + 1];
        double[] v = new double[n + 1];
        int[] p = new int[n + 1];
        int[] way = new int[n + 1];

        for (int i = 1; i <= n; i++) {
            p[0] = i;
            int j0 = 0;
            double[] minv = new double[n + 1];
            boolean[] used = new boolean[n + 1];
            Arrays.fill(minv, Double.POSITIVE_INFINITY);
            Arrays.fill(used, false);
            do {
                used[j0] = true;
                int i0 = p[j0], j1 = 0;
                double delta = Double.POSITIVE_INFINITY;
                for (int j = 1; j <= n; j++) if (!used[j]) {
                    double cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                    if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
                    if (minv[j] < delta) { delta = minv[j]; j1 = j; }
                }
                for (int j = 0; j <= n; j++) {
                    if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                    else { minv[j] -= delta; }
                }
                j0 = j1;
            } while (p[j0] != 0);
            do {
                int j1 = way[j0];
                p[j0] = p[j1];
                j0 = j1;
            } while (j0 != 0);
        }
        double matchWeight = 0;
        for (int j = 1; j <= n; j++) {
            int i = p[j];
            double w = (i <= W.length && j <= W[0].length) ? W[i - 1][j - 1] : 0;
            matchWeight += w;
        }
        return matchWeight;
    }

    // ----------------------------- Utilities -----------------------------
    // lightweight int list
    static final class IntList {
        int[] a = new int[16];
        int sz = 0;
        void add(int v) { if (sz == a.length) a = Arrays.copyOf(a, sz * 2); a[sz++] = v; }
        int size() { return sz; }
        int get(int i) { return a[i]; }
        void clear() { sz = 0; }
        int[] toArray() { return Arrays.copyOf(a, sz); }
    }
}
