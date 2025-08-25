import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

/**
 * Cora ablation driver for L-RMC.
 * <p>
 * - Loads Cora (content + cites)
 * - Calls clique2_ablations.runLaplacianRMC(adj) to get reconstruction snapshots
 * - Scores snapshots with the calibrated surrogate:
 * \tilde S_L(C) = |C| * ( dbar(C) - alpha * sqrt(Q + epsilon) )
 * where Q = d^T L_C d from snapshots, dbar = sumDegIn / |C|
 * - For each (epsilon, alpha) combination, assigns each node the majority label
 * of the highest-scoring snapshot that contains it, then reports accuracy.
 * <p>
 * α choices match the ablation in the paper:
 * α ∈ { diam(C), 1/√λ2(C) }  and  ε ∈ {1e-8, 1e-6, 1e-4}.
 * <p>
 * References to definitions and ablation settings:
 * - Surrogate + α calibration and bounds: Section 5, eqs. (6)-(7), SeL.
 * - Ablation settings on ε and α on Cora: Section 7.3 and Figure 3.
 */
class LRMCablations2 {

    // Ablation grid
    static final double[] EPSILONS = {1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 20000, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e12, 1e13, 1e14};

    enum AlphaKind {DIAM, INV_SQRT_LAMBDA2}

    public static void main(String[] args) throws Exception {
        if (args.length < 3) {
            System.err.println("Usage: java LRMCmkpaper <path/to/cora.content> <path/to/cora.cites> <output_csv>");
            return;
        }
        final Path contentPath = Paths.get(args[0]);
        final Path citesPath = Paths.get(args[1]);
        final Path outCsv = Paths.get(args[2]);
        final Path outSeeds = (args.length >= 4 ? Paths.get(args[3]) : null);
        final AlphaKind alphaKind = (args.length >= 5 ? parseAlpha(args[4]) : AlphaKind.DIAM);
        final double eps = (args.length >= 6 ? Double.parseDouble(args[5]) : 1e-6);

        // 1) Load Cora graph and labels
        GraphData G = loadCora(contentPath, citesPath);
        System.out.printf(Locale.US, "# Loaded Cora: n=%d, m=%d, classes=%d%n",
                G.n, G.m, G.labelNames.length);

        // 2) Run L-RMC reconstruction to get snapshots
        List<clique2_ablations.SnapshotDTO> snaps = clique2_ablations.runLaplacianRMC(G.adj1Based);
        System.out.printf(Locale.US, "# Reconstruction snapshots: %d%n", snaps.size());

        // 3) Evaluate accuracy for every (epsilon, alpha)
        List<ResultRow> outRows = new ArrayList<>();
        for (double eps : EPSILONS) {
            for (AlphaKind aKind : AlphaKind.values()) {
                double acc = labelAccuracyFromSnapshots(snaps, G, eps, aKind);
                String alphaName = (aKind == AlphaKind.DIAM) ? "diam(C)" : "1/sqrt(lambda2)";
                outRows.add(new ResultRow(eps, alphaName, acc, G.n, G.m));
                System.out.printf(Locale.US, "epsilon=%.0e, alpha=%s, accuracy=%.4f%n",
                        eps, alphaName, acc);
            }
        }

        // 4) Save CSV
        try (BufferedWriter w = Files.newBufferedWriter(outCsv, StandardCharsets.UTF_8)) {
            w.write("epsilon,alpha,accuracy,nodes,edges\n");
            for (ResultRow r : outRows) {
                w.write(String.format(Locale.US, "%.0e,%s,%.6f,%d,%d%n",
                        r.epsilon, r.alpha, r.accuracy, r.n, r.m));
            }
        }
        System.out.println("# Wrote: " + outCsv.toAbsolutePath());

        if (outSeeds != null) {
            exportLrmcSeeds(snaps, G, eps, alphaKind, outSeeds);
        }
    }

    // ---------- Accuracy from α-calibrated surrogate over all nodes ----------
    static double labelAccuracyFromSnapshots(
            List<clique2_ablations.SnapshotDTO> snaps,
            GraphData G,
            double epsilon,
            AlphaKind alphaKind
    ) {
        final int n = G.n;
        final int numClasses = G.labelNames.length;

        double[] bestScore = new double[n];
        int[] bestLabel = new int[n];
        Arrays.fill(bestScore, Double.NEGATIVE_INFINITY);
        Arrays.fill(bestLabel, -1);

        // Pre-allocate scratch to avoid churn in inner loops
        boolean[] inC = new boolean[n];

        for (clique2_ablations.SnapshotDTO s : snaps) {
            final int[] nodes = s.nodes;           // 0-based ids
            final int k = nodes.length;
            if (k == 0) continue;

            // Mark current component
            for (int u : nodes) inC[u] = true;

            // Average internal degree in the component
            final double dbar = (k == 0) ? 0.0 : (s.sumDegIn / (double) k);
            // Laplacian energy Q and calibrated α
            final double Q = s.Q;
            double alpha;
            if (alphaKind == AlphaKind.DIAM) {
                alpha = approxDiameter(nodes, G.adj1Based, inC);
            } else {
                double lam2 = approxLambda2(nodes, G.adj1Based, inC);
                if (lam2 <= 1e-12) lam2 = 1e-12; // guard for nC=1 or numerical zeros
                alpha = 1.0 / Math.sqrt(lam2);
            }

            // Calibrated surrogate \tilde S_L(C)
            final double score = k * (dbar - alpha * Math.sqrt(Q + epsilon));

            // Majority label of this component
            int maj = majorityLabel(nodes, G.labels, numClasses);

            // Update node-wise best snapshot under this scoring
            for (int u : nodes) {
                if (score > bestScore[u]) {
                    bestScore[u] = score;
                    bestLabel[u] = maj;
                }
            }

            // Unmark
            for (int u : nodes) inC[u] = false;
        }

        // Compute accuracy over all nodes
        int correct = 0;
        for (int u = 0; u < n; u++) {
            // Every node appears in at least its singleton snapshot
            if (bestLabel[u] == G.labels[u]) correct++;
        }
        return correct / (double) n;
    }

    // ---------- Majority label ----------
    static int majorityLabel(int[] nodes, int[] labels, int numClasses) {
        int[] cnt = new int[numClasses];
        for (int u : nodes) cnt[labels[u]]++;
        int best = 0, arg = 0;
        for (int c = 0; c < numClasses; c++) {
            if (cnt[c] > best) {
                best = cnt[c];
                arg = c;
            }
        }
        return arg;
    }

    // ---------- Approximate diameter by 2-sweep BFS on the induced subgraph ----------
    static int approxDiameter(int[] nodes, List<Integer>[] adj1, boolean[] inC) {
        if (nodes.length <= 1) return 0;

        int start = nodes[0];
        int u = farthestInComponent(start, adj1, inC).node;
        BFSResult r = farthestInComponent(u, adj1, inC);
        return r.dist;
    }

    static class BFSResult {
        final int node, dist;

        BFSResult(int node, int dist) {
            this.node = node;
            this.dist = dist;
        }
    }

    static BFSResult farthestInComponent(int src0, List<Integer>[] adj1, boolean[] inC) {
        int n = inC.length;
        int[] dist = new int[n];
        Arrays.fill(dist, -1);
        ArrayDeque<Integer> q = new ArrayDeque<>();
        dist[src0] = 0;
        q.add(src0);
        int bestNode = src0, bestDist = 0;
        while (!q.isEmpty()) {
            int u = q.poll();
            int du = dist[u];
            if (du > bestDist) {
                bestDist = du;
                bestNode = u;
            }
            for (int v1 : adj1[u + 1]) {
                int v = v1 - 1;
                if (!inC[v]) continue;
                if (dist[v] >= 0) continue;
                dist[v] = du + 1;
                q.add(v);
            }
        }
        return new BFSResult(bestNode, bestDist);
    }

    // ---------- Approximate λ2(L_C) via orthogonalized power iteration ----------
    static double approxLambda2(int[] nodes, List<Integer>[] adj1, boolean[] inC) {
        final int k = nodes.length;
        if (k <= 1) return 0.0;

        // local index mapping
        int[] loc = new int[inC.length];
        Arrays.fill(loc, -1);
        for (int i = 0; i < k; i++) loc[nodes[i]] = i;

        // Build degrees inside C and find d_max
        int[] deg = new int[k];
        int dmax = 0;
        for (int i = 0; i < k; i++) {
            int u = nodes[i];
            int du = 0;
            for (int w1 : adj1[u + 1]) {
                int w = w1 - 1;
                if (loc[w] >= 0) du++;
            }
            deg[i] = du;
            if (du > dmax) dmax = du;
        }
        if (dmax == 0) return 0.0;

        final double c = 2.0 * dmax + 1.0; // shift

        double[] x = new double[k];
        Random rng = new Random(42);
        for (int i = 0; i < k; i++) x[i] = rng.nextDouble() - 0.5;
        orthToOnes(x);
        normalize(x);

        double[] Lx = new double[k];
        double[] y = new double[k];

        final int iters = 30;
        for (int it = 0; it < iters; it++) {
            // L x
            Arrays.fill(Lx, 0.0);
            for (int i = 0; i < k; i++) {
                double sumNbr = 0.0;
                int u = nodes[i];
                for (int w1 : adj1[u + 1]) {
                    int j = loc[w1 - 1];
                    if (j >= 0) sumNbr += x[j];
                }
                Lx[i] = deg[i] * x[i] - sumNbr;
            }
            // y = x - (1/c)*Lx
            for (int i = 0; i < k; i++) y[i] = x[i] - Lx[i] / c;
            orthToOnes(y);
            normalize(y);
            System.arraycopy(y, 0, x, 0, k);
        }

        // Rayleigh quotient x^T L x
        double num = 0.0;
        for (int i = 0; i < k; i++) {
            double sumNbr = 0.0;
            int u = nodes[i];
            for (int w1 : adj1[u + 1]) {
                int j = loc[w1 - 1];
                if (j >= 0) sumNbr += x[j];
            }
            double Lxi = deg[i] * x[i] - sumNbr;
            num += x[i] * Lxi;
        }
        // x is unit norm, so denom ~ 1
        return Math.max(num, 0.0);
    }

    static void orthToOnes(double[] v) {
        double mean = 0.0;
        for (double x : v) mean += x;
        mean /= v.length;
        for (int i = 0; i < v.length; i++) v[i] -= mean;
    }

    static void normalize(double[] v) {
        double nrm2 = 0.0;
        for (double x : v) nrm2 += x * x;
        if (nrm2 <= 0) {
            v[0] = 1.0;
            nrm2 = 1.0;
        }
        double inv = 1.0 / Math.sqrt(nrm2);
        for (int i = 0; i < v.length; i++) v[i] *= inv;
    }

    // ---------- Cora loader ----------
    static GraphData loadCora(Path content, Path cites) throws IOException {
        Map<String, Integer> id2idx = new LinkedHashMap<>();
        Map<String, Integer> lbl2idx = new LinkedHashMap<>();
        List<String> lblNames = new ArrayList<>();
        List<Integer> labelsList = new ArrayList<>();

        // Pass 1: content defines node universe and labels
        try (BufferedReader br = Files.newBufferedReader(content, StandardCharsets.UTF_8)) {
            String s;
            while ((s = br.readLine()) != null) {
                s = s.trim();
                if (s.isEmpty()) continue;
                String[] tok = s.split("\\s+");
                String id = tok[0];
                String lab = tok[tok.length - 1];
                int u = id2idx.computeIfAbsent(id, _k -> id2idx.size());
                int c = lbl2idx.computeIfAbsent(lab, _k -> {
                    lblNames.add(lab);
                    return lblNames.size() - 1;
                });
                // Extend labels list to position u if needed
                while (labelsList.size() <= u) labelsList.add(0);
                labelsList.set(u, c);
            }
        }
        int n = id2idx.size();
        int[] labels = new int[n];
        for (int i = 0; i < n; i++) labels[i] = labelsList.get(i);

        // Temp adjacency as sets to dedup
        @SuppressWarnings("unchecked")
        HashSet<Integer>[] adjSet1 = new HashSet[n + 1];
        for (int i = 1; i <= n; i++) adjSet1[i] = new HashSet<>();

        // Pass 2: cites edges
        long mUndir = 0;
        try (BufferedReader br = Files.newBufferedReader(cites, StandardCharsets.UTF_8)) {
            String s;
            while ((s = br.readLine()) != null) {
                s = s.trim();
                if (s.isEmpty() || s.startsWith("#")) continue;
                String[] tok = s.split("\\s+|,");
                if (tok.length < 2) continue;
                Integer ui = id2idx.get(tok[0]);
                Integer vi = id2idx.get(tok[1]);
                if (ui == null || vi == null) continue; // skip unknown ids
                int a = ui + 1, b = vi + 1;             // to 1-based
                if (a == b) continue;
                if (adjSet1[a].add(b)) {
                    adjSet1[b].add(a);
                    mUndir++;
                }
            }
        }

        @SuppressWarnings("unchecked")
        List<Integer>[] adj1 = new ArrayList[n + 1];
        for (int i = 1; i <= n; i++) {
            adj1[i] = new ArrayList<>(adjSet1[i]);
        }

        GraphData G = new GraphData();
        G.n = n;
        G.m = mUndir;
        G.adj1Based = adj1;
        G.labels = labels;
        G.labelNames = lblNames.toArray(new String[0]);
        return G;
    }


    // Parse alpha kind from string
    static AlphaKind parseAlpha(String s) {
        String t = s.trim().toUpperCase(Locale.ROOT);
        if (t.equals("DIAM") || t.equals("DIAM(C)") || t.equals("DIAMETER")) return AlphaKind.DIAM;
        if (t.equals("INV_SQRT_LAMBDA2") || t.equals("1/SQRT(LAMBDA2)") || t.equals("LAMBDA2"))
            return AlphaKind.INV_SQRT_LAMBDA2;
        throw new IllegalArgumentException("Unknown alpha kind: " + s);
    }

    // Export seeds as a node->cluster partition built from best snapshot per node
    static void exportLrmcSeeds(
            List<clique2_ablations.SnapshotDTO> snaps,
            GraphData G,
            double epsilon,
            AlphaKind alphaKind,
            Path outJson) throws IOException {
        final int n = G.n;
        final boolean[] inC = new boolean[n];

        // 1) Score every snapshot with the calibrated surrogate: k*(dbar - alpha*sqrt(Q+eps))
        final double[] snapScore = new double[snaps.size()];
        Arrays.fill(snapScore, Double.NEGATIVE_INFINITY);
        for (int i = 0; i < snaps.size(); i++) {
            clique2_ablations.SnapshotDTO s = snaps.get(i);
            final int[] nodes = s.nodes; // 0-based ids
            final int k = nodes.length;
            if (k == 0) continue;
            for (int u : nodes) inC[u] = true;
            final double dbar = s.sumDegIn / Math.max(1.0, (double) k);
            final double Q = s.Q;
            final double alpha;
            if (alphaKind == AlphaKind.DIAM) {
                alpha = approxDiameter(nodes, G.adj1Based, inC);
            } else {
                double lam2 = approxLambda2(nodes, G.adj1Based, inC);
                if (lam2 <= 1e-12) lam2 = 1e-12;
                alpha = 1.0 / Math.sqrt(lam2);
            }
            final double score = k * (dbar - alpha * Math.sqrt(Q + epsilon));
            snapScore[i] = score;
            for (int u : nodes) inC[u] = false;
        }

        // 2) Best snapshot per node
        final double[] best = new double[n];
        final int[] bestSnap = new int[n];
        Arrays.fill(best, Double.NEGATIVE_INFINITY);
        Arrays.fill(bestSnap, -1);
        for (int i = 0; i < snaps.size(); i++) {
            clique2_ablations.SnapshotDTO s = snaps.get(i);
            double sc = snapScore[i];
            for (int u : s.nodes) {
                if (sc > best[u]) {
                    best[u] = sc;
                    bestSnap[u] = i;
                }
            }
        }

        // 3) Build cluster membership map: snapshot id -> members
        LinkedHashMap<Integer, List<Integer>> members = new LinkedHashMap<>();
        for (int u = 0; u < n; u++) {
            int sid = bestSnap[u];
            if (sid < 0) continue; // should not happen if singleton snapshots exist
            members.computeIfAbsent(sid, _k -> new ArrayList<>()).add(u);
        }

        // 4) Relabel to contiguous cluster ids
        LinkedHashMap<Integer, Integer> snap2cluster = new LinkedHashMap<>();
        int cid = 0;
        for (int sid : members.keySet()) snap2cluster.put(sid, cid++);

        // 5) Write JSON
        try (BufferedWriter w = Files.newBufferedWriter(outJson, StandardCharsets.UTF_8)) {
            w.write("{\"meta\":{");
            w.write("\"epsilon\":" + epsilon + ",\"alpha_kind\":\"" + alphaKind + "\",\"n\":" + G.n + ",\"m\":" + G.m + "},");
            w.write("\"clusters\":[\n");
            boolean first = true;
            for (Map.Entry<Integer, List<Integer>> e : members.entrySet()) {
                int sid = e.getKey();
                int clusterId = snap2cluster.get(sid);
                clique2_ablations.SnapshotDTO s = snaps.get(sid);
                if (!first) w.write(",\n");
                first = false;
                w.write("  {\"cluster_id\":" + clusterId);
                w.write(",\"snapshot_id\":" + sid);
                w.write(",\"score\":" + snapScore[sid]);
                w.write(",\"k_seed\":" + s.nodes.length);
                w.write(",\"members\":" + intListToJson(e.getValue()));
                w.write(",\"seed_nodes\":" + intArrayToJson(s.nodes));
                w.write("}");
            }
            w.write("\n]}");
        }
        System.out.println("# Wrote seeds: " + outJson.toAbsolutePath());
    }

    static String intArrayToJson(int[] a) {
        StringBuilder sb = new StringBuilder();
        sb.append('[');
        for (int i = 0; i < a.length; i++) {
            if (i > 0) sb.append(',');
            sb.append(a[i]);
        }
        sb.append(']');
        return sb.toString();
    }

    static String intListToJson(List<Integer> a) {
        StringBuilder sb = new StringBuilder();
        sb.append('[');
        for (int i = 0; i < a.size(); i++) {
            if (i > 0) sb.append(',');
            sb.append(a.get(i));
        }
        sb.append(']');
        return sb.toString();
    }


    // ---------- Data holders ----------
    static final class GraphData {
        int n;
        long m;
        List<Integer>[] adj1Based; // 1-based adjacency
        int[] labels;              // 0-based labels per node id
        String[] labelNames;
    }

    static final class ResultRow {
        final double epsilon;
        final String alpha;
        final double accuracy;
        final int n;
        final long m;

        ResultRow(double epsilon, String alpha, double accuracy, int n, long m) {
            this.epsilon = epsilon;
            this.alpha = alpha;
            this.accuracy = accuracy;
            this.n = n;
            this.m = m;
        }
    }
}
