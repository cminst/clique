package UsacoProbs;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

public class LRMCablations {

    // ---------------- existing runtime experiment constants (kept intact) ----------------
    static final int NUM_CLUSTERS = 10;
    static final double CLUSTER_FRACTION = 0.20;
    static final double[] PINTRA_SERIES = {0.010};
    static final double PINTER_FIXED = 1e-4;
    static final String[] SERIES_LABELS = {"p0.010"};
    static final int S_MIN = 10_000;
    static final int S_MAX = 1_100_000;
    static final int NUM_SIZES = 30;
    static final int TRIALS = 3;
    static final double EPSILON = 1e-6;
    static final String EXTRA_HEAP = "-Xmx8g";
    static final boolean PASS_EPSILON = true;
    static final long SEED = 123456789L;

    // ---------------- new: Cora ablations + calibration settings ----------------
    static final double[] EPS_SWEEP = {1e-8, 1e-6, 1e-4};
    static final String[] ALPHAS = {"diam", "invsqrt_lambda2"};
    static final int TOP_K_SEEDS = 100;      // keep fixed across runs for clean comparisons
    static final int MAX_L2_ITERS = 40;      // spectral estimate iterations
    static final boolean EXPORT_SEEDS = true;// write seed lists for each setting

    static String clique2Main;
    static String outputCsvFile;

    public static void main(String[] args) throws Exception {
        if (args.length == 0) {
            System.err.println("Usage:\n" +
                    "  java LRMCmkpaper <CLIQUE2_MAIN> <output_csv_file>\n" +
                    "  or\n" +
                    "  java LRMCmkpaper cora <cora.content> <cora.cites> <ablations_out_csv>\n");
            return;
        }

        if (args[0].equalsIgnoreCase("cora")) {
            if (args.length < 4) {
                System.err.println("Usage: java LRMCmkpaper cora <cora.content> <cora.cites> <out_csv>");
                return;
            }
            String contentPath = args[1];
            String citesPath = args[2];
            String outCsv = args[3];
            runCoraAblations(contentPath, citesPath, outCsv);
            return;
        }

        // default path: keep original runtime benchmarking
        if (args.length < 2) {
            System.err.println("Usage: java LRMCmkpaper <CLIQUE2_MAIN> <output_csv_file>");
            return;
        }
        clique2Main = args[0];
        outputCsvFile = args[1];
        runRuntimeBenchmark();
    }

    // =====================================================
    //                 CORA ABLATIONS PIPELINE
    // =====================================================
    static void runCoraAblations(String contentPath, String citesPath, String outCsv) throws Exception {
        System.out.println("[CORA] Loading dataset...");
        Cora cora = readCora(contentPath, citesPath);
        System.out.printf(Locale.US, "[CORA] nodes=%d edges=%d classes=%d\n", cora.n, cora.m, cora.numClasses);

        // L-RMC reconstruction snapshots (independent of epsilon)
        System.out.println("[CORA] Building degeneracy order and reconstruction snapshots...");
        Reconstruction recon = buildReconstructionSnapshots(cora.adj);
        System.out.printf(Locale.US, "[CORA] captured %d snapshots\n", recon.snaps.size());

        // Evaluate ablations across epsilon and alpha
        Path out = Paths.get(outCsv);
        try (BufferedWriter w = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {
            w.write("epsilon,alpha,K,coverage_pct,accuracy,macro_f1\n");

            for (double eps : EPS_SWEEP) {
                for (String alphaName : ALPHAS) {
                    System.out.printf(Locale.US, "[CORA] eps=%.1e alpha=%s ranking seeds...\n", eps, alphaName);
                    List<RankedSeed> seeds = rankSeeds(cora, recon, eps, alphaName);

                    // choose top K
                    int K = Math.min(TOP_K_SEEDS, seeds.size());
                    List<RankedSeed> topSeeds = new ArrayList<>(seeds.subList(0, K));

                    // export seed lists if requested
                    if (EXPORT_SEEDS) {
                        String base = String.format(Locale.US, "seeds_eps%.0e_%s_K%d.txt", eps, alphaName, K);
                        Path seedFile = out.getParent() == null ? Paths.get(base) : out.getParent().resolve(base);
                        exportSeeds(seedFile, topSeeds, cora);
                    }

                    // compute baseline accuracy by letting seeds vote per node
                    EvalResult er = evaluateComponentMajority(cora, topSeeds);

                    w.write(String.format(Locale.US, "%.3g,%s,%d,%.2f,%.4f,%.4f\n",
                            eps, alphaName, K, 100.0 * er.coverage, er.accuracy, er.macroF1));
                    w.flush();

                    System.out.printf(Locale.US, "[CORA] eps=%.1e alpha=%s K=%d coverage=%.1f%% acc=%.3f macroF1=%.3f\n",
                            eps, alphaName, K, 100.0 * er.coverage, er.accuracy, er.macroF1);
                }
            }
        }
        System.out.println("[CORA] Done. CSV written to: " + out.toAbsolutePath());
    }

    static void exportSeeds(Path path, List<RankedSeed> seeds, Cora cora) throws IOException {
        try (BufferedWriter w = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
            for (int i = 0; i < seeds.size(); i++) {
                RankedSeed rs = seeds.get(i);
                w.write("# seed " + i + ", score=" + rs.score);
                w.newLine();
                int[] nodes = rs.nodes;
                StringBuilder sb = new StringBuilder();
                for (int j = 0; j < nodes.length; j++) {
                    if (j > 0) sb.append(' ');
                    sb.append(nodes[j]); // 0-based indices into cora arrays
                }
                w.write(sb.toString());
                w.newLine();
            }
        }
        System.out.println("[CORA] wrote seeds to: " + path.toAbsolutePath());
    }

    // Rank components by \tilde S_L(C) = nC (dbar - alpha * sqrt(Q + eps))
    static List<RankedSeed> rankSeeds(Cora cora, Reconstruction recon, double eps, String alphaName) {
        List<RankedSeed> out = new ArrayList<>();
        // cache expensive per-snapshot structures
        Map<Integer, LocalSubgraph> gCache = new HashMap<>();

        for (int idx = 0; idx < recon.snaps.size(); idx++) {
            Snapshot s = recon.snaps.get(idx);
            if (s.nC <= 1) continue; // skip singletons

            double alpha;
            if ("diam".equals(alphaName)) {
                LocalSubgraph g = gCache.computeIfAbsent(idx, k -> buildLocal(cora.adj, s.nodes));
                int diam = graphDiameter(g);
                alpha = Math.max(1, diam); // guard
            } else if ("invsqrt_lambda2".equals(alphaName)) {
                LocalSubgraph g = gCache.computeIfAbsent(idx, k -> buildLocal(cora.adj, s.nodes));
                double lam2 = estimateLambda2(g, MAX_L2_ITERS);
                if (lam2 <= 1e-12) continue; // ill-defined, skip
                alpha = 1.0 / Math.sqrt(lam2);
            } else {
                throw new IllegalArgumentException("Unknown alpha: " + alphaName);
            }

            double dbar = s.sumDegIn / (double) s.nC;
            double score = s.nC * (dbar - alpha * Math.sqrt(s.Q + eps));
            out.add(new RankedSeed(s.nodes, score));
        }
        out.sort((a, b) -> Double.compare(b.score, a.score));
        return out;
    }

    static EvalResult evaluateComponentMajority(Cora cora, List<RankedSeed> seeds) {
        int n = cora.n;
        int[] pred = new int[n];
        Arrays.fill(pred, -1);

        // global majority fallback
        int[] labelCounts = new int[cora.numClasses];
        for (int y : cora.labels) labelCounts[y]++;
        int globalMaj = 0;
        for (int c = 1; c < labelCounts.length; c++) if (labelCounts[c] > labelCounts[globalMaj]) globalMaj = c;

        // assign by descending seed score
        for (RankedSeed rs : seeds) {
            int[] nodes = rs.nodes;
            int maj = majorityLabel(nodes, cora.labels, cora.numClasses);
            for (int v : nodes) if (pred[v] == -1) pred[v] = maj;
        }
        // fallback
        int assigned = 0;
        for (int i = 0; i < n; i++) {
            if (pred[i] == -1) pred[i] = globalMaj; else assigned++;
        }

        double acc = 0;
        for (int i = 0; i < n; i++) if (pred[i] == cora.labels[i]) acc++;
        acc /= n;

        double macroF1 = macroF1(pred, cora.labels, cora.numClasses);
        double coverage = assigned / (double) n;
        return new EvalResult(acc, macroF1, coverage);
    }

    static int majorityLabel(int[] nodes, int[] labels, int numClasses) {
        int[] cnt = new int[numClasses];
        for (int v : nodes) cnt[labels[v]]++;
        int best = 0;
        for (int c = 1; c < numClasses; c++) if (cnt[c] > cnt[best]) best = c;
        return best;
    }

    static double macroF1(int[] pred, int[] truth, int C) {
        double f1sum = 0;
        for (int c = 0; c < C; c++) {
            int tp = 0, fp = 0, fn = 0;
            for (int i = 0; i < pred.length; i++) {
                boolean p = pred[i] == c;
                boolean t = truth[i] == c;
                if (p && t) tp++; else if (p) fp++; else if (t) fn++;
            }
            double prec = tp + fp == 0 ? 0 : (double) tp / (tp + fp);
            double rec  = tp + fn == 0 ? 0 : (double) tp / (tp + fn);
            double f1   = prec + rec == 0 ? 0 : 2 * prec * rec / (prec + rec);
            f1sum += f1;
        }
        return f1sum / C;
    }

    // ---------------- Reconstruction snapshots via degeneracy-peeling order ----------------
    static Reconstruction buildReconstructionSnapshots(int[][] adj) {
        int n = adj.length;
        int[] deg = new int[n];
        for (int i = 0; i < n; i++) deg[i] = adj[i].length;

        // priority-queue peeling to get a degeneracy-style order
        boolean[] removed = new boolean[n];
        PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.<int[]>comparingInt(a -> a[0]).thenComparingInt(a -> a[1]));
        for (int i = 0; i < n; i++) pq.offer(new int[]{deg[i], i});
        int[] order = new int[n];
        int oi = 0;
        while (!pq.isEmpty()) {
            int[] top = pq.poll();
            int d = top[0], u = top[1];
            if (removed[u] || d != deg[u]) continue;
            removed[u] = true;
            order[oi++] = u;
            for (int v : adj[u]) if (!removed[v]) { deg[v]--; pq.offer(new int[]{deg[v], v}); }
        }
        int[] addOrder = new int[n];
        for (int i = 0; i < n; i++) addOrder[i] = order[n - 1 - i];

        // reconstruct with DSU and track component at each activation
        DSU dsu = new DSU(n);
        boolean[] active = new boolean[n];
        int[] degIn = new int[n];
        @SuppressWarnings("unchecked")
        HashSet<Integer>[] compNodes = new HashSet[n];
        for (int i = 0; i < n; i++) compNodes[i] = new HashSet<>();

        Reconstruction recon = new Reconstruction();

        for (int u : addOrder) {
            active[u] = true;
            compNodes[u].add(u);
            // connect to active neighbors
            for (int v : adj[u]) if (active[v]) {
                degIn[u]++; degIn[v]++;
                int r1 = dsu.find(u), r2 = dsu.find(v);
                if (r1 != r2) {
                    int r = dsu.union(r1, r2);
                    int o = (r == r1) ? r2 : r1;
                    if (r != o) { compNodes[r].addAll(compNodes[o]); compNodes[o].clear(); }
                }
            }
            int r = dsu.find(u);
            int[] nodes = compNodes[r].stream().mapToInt(x -> x).toArray();
            Arrays.sort(nodes);
            // compute Q and sumDegIn for this snapshot
            double Q = 0;
            long sumDeg = 0;
            HashSet<Integer> set = new HashSet<>();
            for (int x : nodes) set.add(x);
            for (int x : nodes) {
                int dx = degIn[x];
                sumDeg += dx;
                for (int y : adj[x]) if (y > x && set.contains(y)) {
                    int dy = degIn[y];
                    double d = dx - dy;
                    Q += d * d;
                }
            }
            Snapshot snap = new Snapshot(nodes, nodes.length, sumDeg, Q);
            recon.snaps.add(snap);
        }
        return recon;
    }

    // ---------------- Local subgraph utilities ----------------
    static LocalSubgraph buildLocal(int[][] globalAdj, int[] nodes) {
        int n = nodes.length;
        int[] map = new int[globalAdj.length];
        Arrays.fill(map, -1);
        for (int i = 0; i < n; i++) map[nodes[i]] = i;
        List<Integer>[] lst = new ArrayList[n];
        for (int i = 0; i < n; i++) lst[i] = new ArrayList<>();
        int[] deg = new int[n];
        for (int gi = 0; gi < n; gi++) {
            int g = nodes[gi];
            for (int nb : globalAdj[g]) {
                int li = map[nb];
                if (li >= 0) { lst[gi].add(li); deg[gi]++; }
            }
        }
        int[][] adj = new int[n][];
        for (int i = 0; i < n; i++) {
            adj[i] = new int[lst[i].size()];
            for (int j = 0; j < lst[i].size(); j++) adj[i][j] = lst[i].get(j);
        }
        return new LocalSubgraph(n, adj, deg);
    }

    static int graphDiameter(LocalSubgraph g) {
        if (g.n == 0) return 0;
        int u = 0;
        int[] dist = bfs(g, u);
        int fur = u;
        for (int i = 0; i < g.n; i++) if (dist[i] > dist[fur]) fur = i;
        dist = bfs(g, fur);
        int diam = 0;
        for (int d : dist) if (d > diam) diam = d;
        return diam;
    }

    static int[] bfs(LocalSubgraph g, int src) {
        int[] d = new int[g.n];
        Arrays.fill(d, -1);
        ArrayDeque<Integer> q = new ArrayDeque<>();
        q.add(src); d[src] = 0;
        while (!q.isEmpty()) {
            int u = q.poll();
            for (int v : g.adj[u]) if (d[v] == -1) { d[v] = d[u] + 1; q.add(v); }
        }
        return d;
    }

    // Estimate lambda2(L) by power iteration on P = I - tau L with projection
    static double estimateLambda2(LocalSubgraph g, int iters) {
        if (g.n <= 1) return 0.0;
        int n = g.n;
        int maxDeg = 0; for (int d : g.deg) if (d > maxDeg) maxDeg = d;
        double tau = 1.0 / (maxDeg + 1.0);

        double[] v = new double[n];
        Random rng = new Random(42);
        for (int i = 0; i < n; i++) v[i] = rng.nextDouble() - 0.5;
        projectMeanZero(v);
        normalize(v);

        double[] w = new double[n];
        for (int it = 0; it < iters; it++) {
            laplacianTimes(g, v, w);      // w = L v
            for (int i = 0; i < n; i++) w[i] = v[i] - tau * w[i]; // w = (I - tau L) v
            projectMeanZero(w);
            normalize(w);
            System.arraycopy(w, 0, v, 0, n);
        }
        // Rayleigh quotient v^T L v / v^T v (v has unit norm)
        laplacianTimes(g, v, w);
        double num = 0.0;
        for (int i = 0; i < n; i++) num += v[i] * w[i];
        return Math.max(0.0, num);
    }

    static void laplacianTimes(LocalSubgraph g, double[] x, double[] out) {
        Arrays.fill(out, 0.0);
        for (int i = 0; i < g.n; i++) {
            double s = g.deg[i] * x[i];
            for (int j : g.adj[i]) s -= x[j];
            out[i] = s;
        }
    }

    static void projectMeanZero(double[] v) {
        double mean = 0; for (double z : v) mean += z; mean /= v.length;
        for (int i = 0; i < v.length; i++) v[i] -= mean;
    }

    static void normalize(double[] v) {
        double s2 = 0; for (double z : v) s2 += z * z; s2 = Math.sqrt(Math.max(1e-18, s2));
        for (int i = 0; i < v.length; i++) v[i] /= s2;
    }

    // ---------------- Data containers ----------------
    static final class Reconstruction {
        final List<Snapshot> snaps = new ArrayList<>();
    }
    static final class Snapshot {
        final int[] nodes; // 0-based ids in the original graph
        final int nC;
        final long sumDegIn; // sum of internal degrees at this snapshot
        final double Q;      // d^T L_C d at this snapshot
        Snapshot(int[] nodes, int nC, long sumDegIn, double Q) { this.nodes = nodes; this.nC = nC; this.sumDegIn = sumDegIn; this.Q = Q; }
    }
    static final class RankedSeed {
        final int[] nodes; final double score;
        RankedSeed(int[] nodes, double score) { this.nodes = nodes; this.score = score; }
    }
    static final class LocalSubgraph {
        final int n; final int[][] adj; final int[] deg;
        LocalSubgraph(int n, int[][] adj, int[] deg) { this.n = n; this.adj = adj; this.deg = deg; }
    }
    static final class EvalResult {
        final double accuracy, macroF1, coverage;
        EvalResult(double accuracy, double macroF1, double coverage) { this.accuracy = accuracy; this.macroF1 = macroF1; this.coverage = coverage; }
    }

    static final class Cora {
        final int n; final int m; final int[][] adj; final int[] labels; final int numClasses;
        Cora(int n, int m, int[][] adj, int[] labels, int numClasses) { this.n = n; this.m = m; this.adj = adj; this.labels = labels; this.numClasses = numClasses; }
    }

    // =====================================================
    //                   CORA LOADING
    // =====================================================
    static Cora readCora(String contentPath, String citesPath) throws IOException {
        // content: paper_id f1 ... f1433 label
        Map<String, Integer> id2idx = new HashMap<>();
        List<String> labelsStr = new ArrayList<>();
        List<Integer> y = new ArrayList<>();
        Map<String, Integer> label2id = new HashMap<>();
        try (BufferedReader br = Files.newBufferedReader(Paths.get(contentPath), StandardCharsets.UTF_8)) {
            String s;
            int idx = 0;
            while ((s = br.readLine()) != null) {
                if (s.isEmpty()) continue;
                String[] parts = s.trim().split("\\s+");
                String pid = parts[0];
                String lab = parts[parts.length - 1];
                id2idx.put(pid, idx++);
                Integer lid = label2id.get(lab);
                if (lid == null) { lid = label2id.size(); label2id.put(lab, lid); }
                y.add(lid);
                labelsStr.add(lab);
            }
        }
        int n = id2idx.size();
        int[] labels = new int[n];
        for (int i = 0; i < n; i++) labels[i] = y.get(i);

        // cites: src dst
        List<int[]> edges = new ArrayList<>();
        int missing = 0;
        try (BufferedReader br = Files.newBufferedReader(Paths.get(citesPath), StandardCharsets.UTF_8)) {
            String s;
            while ((s = br.readLine()) != null) {
                if (s.isEmpty()) continue;
                String[] parts = s.trim().split("\\s+");
                if (parts.length < 2) continue;
                Integer u = id2idx.get(parts[0]);
                Integer v = id2idx.get(parts[1]);
                if (u == null || v == null) { missing++; continue; }
                if (!u.equals(v)) edges.add(new int[]{u, v});
            }
        }
        // undirected unique
        Collections.sort(edges, (a, b) -> a[0] != b[0] ? Integer.compare(a[0], b[0]) : Integer.compare(a[1], b[1]));
        List<int[]> und = new ArrayList<>();
        int lastU = -1, lastV = -1;
        for (int[] e : edges) {
            int u = Math.min(e[0], e[1]);
            int v = Math.max(e[0], e[1]);
            if (u == lastU && v == lastV) continue;
            und.add(new int[]{u, v});
            lastU = u; lastV = v;
        }
        int m = und.size();
        int[] deg = new int[n];
        for (int[] e : und) { deg[e[0]]++; deg[e[1]]++; }
        int[][] adj = new int[n][];
        for (int i = 0; i < n; i++) adj[i] = new int[deg[i]];
        int[] cur = new int[n];
        for (int[] e : und) {
            int u = e[0], v = e[1];
            adj[u][cur[u]++] = v;
            adj[v][cur[v]++] = u;
        }
        System.out.printf(Locale.US, "[CORA] missing cite endpoints: %d\n", missing);
        return new Cora(n, m, adj, labels, label2id.size());
    }

    // =====================================================
    //             ORIGINAL RUNTIME BENCHMARK PATH
    // =====================================================
    static void runRuntimeBenchmark() throws Exception {
        Random rng = new Random(SEED);
        List<Row> allRows = new ArrayList<>();
        int[] sizes = logSpaced(S_MIN, S_MAX, NUM_SIZES);

        for (int si = 0; si < PINTRA_SERIES.length; si++) {
            String series = SERIES_LABELS[si];
            double pIntra = PINTRA_SERIES[si];
            double pInter = PINTER_FIXED;

            for (int n : sizes) {
                Path inputFile = Files.createTempFile("clique2_input_" + series + "_n" + n + "_", ".txt");
                inputFile.toFile().deleteOnExit();

                long m = generateClusteredGraphToFile(
                        n, NUM_CLUSTERS, CLUSTER_FRACTION,
                        pIntra, pInter, rng, inputFile);

                int kDeg = computeDegeneracyFromFile(n, m, inputFile);
                double theoX = (n + m) * Math.log(Math.max(2, n)) + (double) m * kDeg;

                for (int t = 0; t < TRIALS; t++) {
                    double ms = runClique2(EPSILON, inputFile);
                    allRows.add(new Row(series, n, m, t + 1, ms, theoX, pIntra, pInter));
                }
            }
        }

        double num = 0, den = 0;
        for (Row r : allRows) { num += r.theoX * r.ms; den += r.theoX * r.theoX; }
        double scale = den == 0 ? 0 : num / den;

        try (BufferedWriter writer = Files.newBufferedWriter(Paths.get(outputCsvFile), StandardCharsets.UTF_8)) {
            writer.write("series,n,m,trial,ms,theo_x,normalized_theory_ms,p_intra,p_inter\n");
            for (Row r : allRows) {
                double norm = scale * r.theoX;
                writer.write(String.format(Locale.US, "%s,%d,%d,%d,%.3f,%.3f,%.3f,%.6f,%.6g\n",
                        r.series, r.n, r.m, r.trial, r.ms, r.theoX, norm, r.pIntra, r.pInter));
            }
        }

        Map<String, Map<Integer, List<Row>>> bySeriesSize = new TreeMap<>();
        for (Row r : allRows) {
            bySeriesSize.computeIfAbsent(r.series, s -> new TreeMap<>())
                    .computeIfAbsent(r.n, _k -> new ArrayList<>()).add(r);
        }
        for (var eSeries : bySeriesSize.entrySet()) {
            String s = eSeries.getKey();
            for (var e : eSeries.getValue().entrySet()) {
                int n = e.getKey();
                long m = e.getValue().get(0).m;
                double[] arr = e.getValue().stream().mapToDouble(rr -> rr.ms).toArray();
                double mean = mean(arr), sd = stddev(arr, mean);
                double theoX = e.getValue().get(0).theoX;
                double norm = scale * theoX;
                double pIntra = e.getValue().get(0).pIntra;
                double pInter = e.getValue().get(0).pInter;
                System.out.printf(Locale.US, "# summary,%s,%d,%d,%.3f,%.3f,%.3f,%.6f,%.6g%n",
                        s, n, m, mean, theoX, norm, pIntra, pInter);
                if (TRIALS > 1) {
                    System.out.printf(Locale.US, "# summary_std,%s,%d,%d,%.3f%n", s, n, m, sd);
                }
            }
        }
    }

    // ------------ run clique2 ------------
    private static double runClique2(double epsilon, Path inputFile) throws IOException, InterruptedException {
        String javaBin = System.getProperty("java.home") + File.separator + "bin" + File.separator + "java";
        String classpath = System.getProperty("java.class.path");
        List<String> cmd = new ArrayList<>();
        cmd.add(javaBin); cmd.add(EXTRA_HEAP); cmd.add("-cp"); cmd.add(classpath); cmd.add(clique2Main);
        if (PASS_EPSILON) cmd.add(Double.toString(epsilon));
        cmd.add(inputFile.toAbsolutePath().toString());
        ProcessBuilder pb = new ProcessBuilder(cmd);
        pb.redirectErrorStream(true);
        Process p = pb.start();
        String lastRuntime = null;
        try (BufferedReader br = new BufferedReader(new InputStreamReader(p.getInputStream(), StandardCharsets.UTF_8))) {
            String line; while ((line = br.readLine()) != null) { if (line.startsWith("Runtime:")) lastRuntime = line; }
        }
        int exit = p.waitFor();
        if (exit != 0) throw new RuntimeException("clique2 exited with code " + exit);
        if (lastRuntime == null) throw new RuntimeException("No 'Runtime: ... ms' line from clique2");
        String msStr = lastRuntime.replace("Runtime:", "").replace("ms", "").trim();
        System.out.println(inputFile.toAbsolutePath().toString());
        return Double.parseDouble(msStr);
    }

    // ------------ clustered generator (unchanged) ------------
    static long generateClusteredGraphToFile(
            int n, int k, double frac, double pIntra, double pInter, Random rng, Path outFile) throws IOException {
        int clusterTotal = (int) Math.round(frac * n);
        int[] nodes = new int[n];
        for (int i = 0; i < n; i++) nodes[i] = i + 1;
        shuffle(nodes, rng);
        int base = clusterTotal / k, rem = clusterTotal % k;
        int[][] clusters = new int[k][];
        int idx = 0;
        for (int i = 0; i < k; i++) {
            int sz = base + (i < rem ? 1 : 0);
            clusters[i] = Arrays.copyOfRange(nodes, idx, idx + sz);
            Arrays.sort(clusters[i]);
            idx += sz;
        }
        int[] background = Arrays.copyOfRange(nodes, idx, n);
        Arrays.sort(background);
        Path tmpEdges = Files.createTempFile("edges_only_", ".txt");
        tmpEdges.toFile().deleteOnExit();
        long m = 0;
        try (BufferedWriter w = Files.newBufferedWriter(tmpEdges, StandardCharsets.UTF_8)) {
            for (int i = 0; i < k; i++) m += triPairsToWriter(clusters[i], pIntra, w, rng);
            for (int i = 0; i < k; i++) for (int j = i + 1; j < k; j++) m += rectPairsToWriter(clusters[i], clusters[j], pInter, w, rng);
            for (int i = 0; i < k; i++) m += rectPairsToWriter(clusters[i], background, pInter, w, rng);
            m += triPairsToWriter(background, pInter, w, rng);
        }
        try (BufferedWriter hdr = Files.newBufferedWriter(outFile, StandardCharsets.UTF_8, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)) {
            hdr.write(n + " " + m); hdr.newLine();
        }
        try (OutputStream out = Files.newOutputStream(outFile, StandardOpenOption.APPEND); InputStream in = Files.newInputStream(tmpEdges)) {
            byte[] buf = new byte[1 << 20]; int len; while ((len = in.read(buf)) != -1) out.write(buf, 0, len);
        }
        return m;
    }

    // ------------ degeneracy from file (unchanged) ------------
    static int computeDegeneracyFromFile(int n, long m, Path edgeListFile) throws IOException {
        int[] deg = new int[n];
        try (BufferedReader br = Files.newBufferedReader(edgeListFile, StandardCharsets.UTF_8)) {
            br.readLine(); String s; while ((s = br.readLine()) != null) {
                if (s.isEmpty()) continue; int sp = s.indexOf(' '); if (sp <= 0) continue;
                int u = Integer.parseInt(s.substring(0, sp)) - 1; int v = Integer.parseInt(s.substring(sp + 1)) - 1;
                if (u == v) continue; if (u < 0 || u >= n || v < 0 || v >= n) continue; deg[u]++; deg[v]++; }
        }
        int maxDeg = 0; long totalAdj = 0; for (int d : deg) { if (d > maxDeg) maxDeg = d; totalAdj += d; }
        if (totalAdj > Integer.MAX_VALUE) { int ub = 0; for (int d : deg) if (d > ub) ub = d; return ub; }
        int[] off = new int[n + 1]; for (int i = 0; i < n; i++) off[i + 1] = off[i] + deg[i];
        int[] adj = new int[(int) totalAdj]; int[] cur = Arrays.copyOf(off, off.length);
        try (BufferedReader br = Files.newBufferedReader(edgeListFile, StandardCharsets.UTF_8)) {
            br.readLine(); String s; while ((s = br.readLine()) != null) {
                if (s.isEmpty()) continue; int sp = s.indexOf(' '); if (sp <= 0) continue;
                int u = Integer.parseInt(s.substring(0, sp)) - 1; int v = Integer.parseInt(s.substring(sp + 1)) - 1;
                if (u == v || u < 0 || u >= n || v < 0 || v >= n) continue; adj[cur[u]++] = v; adj[cur[v]++] = u; }
        }
        int[] degree = Arrays.copyOf(deg, deg.length);
        int[] bin = new int[maxDeg + 1]; for (int d : degree) bin[d]++;
        int start = 0; for (int d = 0; d <= maxDeg; d++) { int count = bin[d]; bin[d] = start; start += count; }
        int[] vert = new int[n]; int[] pos = new int[n];
        for (int v = 0; v < n; v++) { pos[v] = bin[degree[v]]; vert[pos[v]] = v; bin[degree[v]]++; }
        for (int d = maxDeg; d > 0; d--) bin[d] = bin[d - 1]; bin[0] = 0;
        int kDeg = 0;
        for (int i = 0; i < n; i++) {
            int v = vert[i]; int dv = degree[v]; if (dv > kDeg) kDeg = dv;
            for (int p = off[v]; p < off[v + 1]; p++) { int u = adj[p]; if (degree[u] > dv) {
                int du = degree[u]; int pu = pos[u]; int pw = bin[du]; int w = vert[pw]; if (u != w) { vert[pu] = w; pos[w] = pu; vert[pw] = u; pos[u] = pw; }
                bin[du]++; degree[u] = du - 1; } }
            degree[v] = 0; }
        return kDeg;
    }

    // ------------ random graph helpers (unchanged) ------------
    static long triPairsToWriter(int[] set, double p, BufferedWriter w, Random rng) throws IOException {
        int s = set.length; if (s < 2 || p <= 0) return 0L; final double logq = Math.log(1.0 - p);
        long written = 0; int row = 0, off = -1; while (row < s - 1) {
            double r = rng.nextDouble(); int skip = (int) Math.floor(Math.log(1.0 - r) / logq);
            off += 1 + skip; while (row < s - 1 && off >= (s - row - 1)) { off -= (s - row - 1); row++; }
            if (row < s - 1) { int u = set[row], v = set[row + 1 + off]; w.write(u + " " + v); w.newLine(); written++; }
        } return written;
    }
    static long rectPairsToWriter(int[] A, int[] B, double p, BufferedWriter w, Random rng) throws IOException {
        int a = A.length, b = B.length; if (a == 0 || b == 0 || p <= 0) return 0L; final double logq = Math.log(1.0 - p);
        long total = 1L * a * b, t = -1, written = 0; while (true) {
            double r = rng.nextDouble(); long skip = (long) Math.floor(Math.log(1.0 - r) / logq);
            t += 1 + skip; if (t >= total) break; int i = (int) (t / b), j = (int) (t % b);
            w.write(A[i] + " " + B[j]); w.newLine(); written++; } return written;
    }

    // ---------------- helpers (unchanged) ----------------
    static int[] logSpaced(int lo, int hi, int k) {
        double a = Math.log(lo), b = Math.log(hi); int[] out = new int[k];
        for (int i = 0; i < k; i++) { double t = i / (double) (k - 1); out[i] = (int) Math.round(Math.exp(a + t * (b - a))); out[i] = Math.max(lo, Math.min(hi, (out[i] + 500) / 1000 * 1000)); }
        for (int i = 1; i < k; i++) if (out[i] <= out[i - 1]) out[i] = out[i - 1] + 1000; out[k - 1] = hi; return out;
    }
    static void shuffle(int[] a, Random rng) { for (int i = a.length - 1; i > 0; i--) { int j = rng.nextInt(i + 1); int t = a[i]; a[i] = a[j]; a[j] = t; } }
    static double mean(double[] x) { double s = 0; for (double v : x) s += v; return s / x.length; }
    static double stddev(double[] x, double mean) { if (x.length <= 1) return 0; double s2 = 0; for (double v : x) { double d = v - mean; s2 += d * d; } return Math.sqrt(s2 / (x.length - 1)); }

    // ---------------- tiny DSU ----------------
    static final class DSU {
        int[] p, sz;
        DSU(int n) { p = new int[n]; sz = new int[n]; for (int i = 0; i < n; i++) { p[i] = i; sz[i] = 1; } }
        int find(int x) { while (p[x] != x) { p[x] = p[p[x]]; x = p[x]; } return x; }
        int union(int a, int b) { int ra = find(a), rb = find(b); if (ra == rb) return ra; if (sz[ra] < sz[rb]) { int t = ra; ra = rb; rb = t; } p[rb] = ra; sz[ra] += sz[rb]; return ra; }
    }

    static final class Row {
        final String series; final int n; final long m; final int trial; final double ms; final double theoX; final double pIntra; final double pInter;
        Row(String series, int n, long m, int trial, double ms, double theoX, double pIntra, double pInter) {
            this.series = series; this.n = n; this.m = m; this.trial = trial; this.ms = ms; this.theoX = theoX; this.pIntra = pIntra; this.pInter = pInter;
        }
    }
}
