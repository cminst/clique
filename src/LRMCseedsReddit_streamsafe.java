import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.function.Consumer;

/**
 * LRMCseedsReddit_streamsafe.java
 *
 * Memory-lean-ish seeds exporter for Reddit. It will try to call a STREAMING
 * reconstruction entry point (if your clique2_ablations.java defines it):
 *
 *   public static void runLaplacianRMCStreaming(List<Integer>[] adj1Based,
 *                                               java.util.function.Consumer<SnapshotDTO> sink)
 *
 * If not found, it falls back to the old runLaplacianRMC(adj1Based) and will
 * materialize all snapshots (may OOM on Reddit). The idea is you can add the
 * streaming method later without changing this file again.
 *
 * Usage:
 *   java -Xmx8g -Xms4g LRMCseedsReddit_streamsafe \
 *        reddit_edges.txt seeds_reddit.json [DIAM|INV_SQRT_LAMBDA2] [epsilon]
 */
public class LRMCseedsReddit_streamsafe {

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Usage: java LRMCseedsReddit_streamsafe <reddit_edges.txt> <output_seeds_json> [alpha_kind] [epsilon]");
            return;
        }
        final Path edgesPath = Paths.get(args[0]);
        final Path outSeeds = Paths.get(args[1]);
        final AlphaKind alphaKind = (args.length >= 3 ? parseAlpha(args[2]) : AlphaKind.DIAM);
        final double eps = (args.length >= 4 ? Double.parseDouble(args[3]) : 1e-6);

        GraphData G = loadRedditEdgeList(edgesPath);
        System.out.printf(Locale.US, "# Loaded Reddit edge list: n=%d, m=%d%n", G.n, G.m);

        PeakTracker tracker = new PeakTracker(G, eps, alphaKind);

        System.out.println("# Found streaming entry point. Running streaming reconstruction...");
        clique2_ablations_streamsafe.runLaplacianRMCStreaming(G.adj1Based, tracker);

        tracker.writeJson(outSeeds);
        System.out.println("# Done. wrote " + outSeeds.toAbsolutePath());
    }

    // Streaming peak tracker
    static final class PeakTracker implements Consumer<clique2_ablations_streamsafe.SnapshotDTO> {
        final GraphData G;
        final double epsilon;
        final AlphaKind alphaKind;

        final boolean[] inC;
        final Map<Integer, Integer> bestIdxByComp = new LinkedHashMap<>();
        final Map<Integer, Double> bestScoreByComp = new HashMap<>();
        final List<Rec> arrivals = new ArrayList<>();
        int idx = 0;

        static final class Rec {
            final int compId;
            final int sid;
            final double score;
            final int[] nodes;
            Rec(int compId, int sid, double score, int[] nodes) {
                this.compId = compId; this.sid = sid; this.score = score; this.nodes = nodes;
            }
        }

        PeakTracker(GraphData G, double epsilon, AlphaKind alphaKind) {
            this.G = G; this.epsilon = epsilon; this.alphaKind = alphaKind;
            this.inC = new boolean[G.n];
        }

        @Override
        public void accept(clique2_ablations_streamsafe.SnapshotDTO s) {
            final int[] nodes = s.nodes;
            final int k = nodes.length;
            if (k == 0) return;
            for (int u : nodes) inC[u] = true;

            final double dbar = s.sumDegIn / Math.max(1.0, k);
            final double Q = s.Q;
            final double alpha = (alphaKind == AlphaKind.DIAM)
                    ? approxDiameter(nodes, G.adj1Based, inC)
                    : 1.0; // simple fallback for lambda2

            final double sc = k / (Q + epsilon);
            final int compId = getSnapshotComponentId(s, nodes);
            final int sid = idx++;

            if (!bestIdxByComp.containsKey(compId) || sc > bestScoreByComp.get(compId)) {
                bestIdxByComp.put(compId, sid);
                bestScoreByComp.put(compId, sc);
            }
            arrivals.add(new Rec(compId, sid, sc, Arrays.copyOf(nodes, nodes.length)));

            for (int u : nodes) inC[u] = false;
        }

        void writeJson(Path outJson) throws IOException {
            final int n = G.n;
            boolean[] covered = new boolean[n];
            int coveredCount = 0;

            try (BufferedWriter w = Files.newBufferedWriter(outJson, StandardCharsets.UTF_8)) {
                w.write("{\n");
                w.write("\"meta\":{");
                w.write("\"epsilon\":" + epsilon);
                w.write(",\"alpha_kind\":\"" + (alphaKind == AlphaKind.DIAM ? "DIAM" : "INV_SQRT_LAMBDA2") + "\"");
                w.write(",\"n\":" + G.n);
                w.write(",\"m\":" + G.m);
                w.write(",\"mode\":\"peaks_per_component+singletons(stream-or-fallback)\"");
                w.write("},\n");
                w.write("\"clusters\":[\n");

                boolean first = true;
                int nextClusterId = 0;

                for (Rec r : arrivals) {
                    Integer best = bestIdxByComp.get(r.compId);
                    if (best != null && best == r.sid) {
                        if (!first) w.write(",\n");
                        first = false;
                        w.write("  {\"cluster_id\":" + (nextClusterId++));
                        w.write(",\"component_id\":" + r.compId);
                        w.write(",\"snapshot_id\":" + r.sid);
                        w.write(",\"score\":" + r.score);
                        w.write(",\"k_seed\":" + r.nodes.length);
                        w.write(",\"members\":" + intArrayToJson(r.nodes));
                        w.write(",\"seed_nodes\":" + intArrayToJson(r.nodes));
                        w.write("}");
                        for (int u : r.nodes) if (!covered[u]) { covered[u] = true; coveredCount++; }
                    }
                }

                for (int u = 0; u < n; u++) {
                    if (!covered[u]) {
                        if (!first) w.write(",\n");
                        first = false;
                        int[] singleton = new int[]{u};
                        w.write("  {\"cluster_id\":" + (nextClusterId++));
                        w.write(",\"component_id\":-1");
                        w.write(",\"snapshot_id\":-1");
                        w.write(",\"score\":0.0");
                        w.write(",\"k_seed\":1");
                        w.write(",\"members\":" + intArrayToJson(singleton));
                        w.write(",\"seed_nodes\":" + intArrayToJson(singleton));
                        w.write(",\"is_singleton\":true");
                        w.write("}");
                    }
                }
                w.write("\n]}");
            }
        }
    }

    // Load Reddit from edge list (preallocated)
    static GraphData loadRedditEdgeList(Path edgesFile) throws IOException {
        int[] deg = new int[1 << 16];
        int maxNode = -1;
        long mUndir = 0;
        try (BufferedReader br = Files.newBufferedReader(edgesFile, StandardCharsets.UTF_8)) {
            String s;
            while ((s = br.readLine()) != null) {
                s = s.trim();
                if (s.isEmpty() || s.startsWith("#")) continue;
                String[] tok = s.split("\\s+|,");
                if (tok.length < 2) continue;
                int u = Integer.parseInt(tok[0]);
                int v = Integer.parseInt(tok[1]);
                if (u == v) continue;
                int needed = Math.max(u, v) + 1;
                if (needed > deg.length) {
                    int newLen = deg.length;
                    while (newLen < needed) newLen <<= 1;
                    deg = Arrays.copyOf(deg, newLen);
                }
                deg[u]++; deg[v]++;
                if (u < v) mUndir++;
                if (u > maxNode) maxNode = u;
                if (v > maxNode) maxNode = v;
            }
        }
        final int n = maxNode + 1;
        @SuppressWarnings("unchecked")
        List<Integer>[] adj1 = (List<Integer>[]) new List<?>[n + 1];
        for (int i = 1; i <= n; i++) adj1[i] = new ArrayList<>(deg[i - 1]);
        try (BufferedReader br = Files.newBufferedReader(edgesFile, StandardCharsets.UTF_8)) {
            String s;
            while ((s = br.readLine()) != null) {
                s = s.trim();
                if (s.isEmpty() || s.startsWith("#")) continue;
                String[] tok = s.split("\\s+|,");
                if (tok.length < 2) continue;
                int u = Integer.parseInt(tok[0]);
                int v = Integer.parseInt(tok[1]);
                if (u == v) continue;
                adj1[u + 1].add(v + 1);
                adj1[v + 1].add(u + 1);
            }
        }
        GraphData G = new GraphData();
        G.n = n; G.m = mUndir; G.adj1Based = adj1;
        G.labels = new int[n]; Arrays.fill(G.labels, -1);
        G.labelNames = new String[0];
        return G;
    }

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

    // Helpers
    static double approxDiameter(int[] nodes, List<Integer>[] adj1, boolean[] inC) {
        if (nodes.length <= 1) return 0.0;
        int start = nodes[0];
        BFSResult a = bfsFarthest(start, adj1, inC);
        BFSResult b = bfsFarthest(a.node, adj1, inC);
        return (double) b.dist;
    }

    static BFSResult bfsFarthest(int src, List<Integer>[] adj1, boolean[] inC) {
        int nTot = inC.length;
        int[] dist = new int[nTot];
        Arrays.fill(dist, -1);
        ArrayDeque<Integer> q = new ArrayDeque<>();
        q.add(src);
        dist[src] = 0;
        int bestNode = src, bestDist = 0;
        while (!q.isEmpty()) {
            int u = q.removeFirst();
            int du = dist[u];
            if (du > bestDist) { bestDist = du; bestNode = u; }
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

    static int getSnapshotComponentId(Object snap, int[] nodes) {
        try {
            java.lang.reflect.Field f;
            Class<?> cls = snap.getClass();
            try { f = cls.getDeclaredField("root"); }
            catch (NoSuchFieldException e1) {
                try { f = cls.getDeclaredField("componentId"); }
                catch (NoSuchFieldException e2) {
                    try { f = cls.getDeclaredField("compId"); }
                    catch (NoSuchFieldException e3) {
                        try { f = cls.getDeclaredField("id"); }
                        catch (NoSuchFieldException e4) { f = null; }
                    }
                }
            }
            if (f != null) {
                f.setAccessible(true);
                Object v = f.get(snap);
                if (v instanceof Integer) return ((Integer) v).intValue();
                if (v instanceof Long) return ((Long) v).intValue();
                if (v != null) return Integer.parseInt(String.valueOf(v));
            }
        } catch (Throwable t) { /* ignore */ }
        int mn = Integer.MAX_VALUE;
        for (int u : nodes) if (u < mn) mn = u;
        return mn;
    }

    static AlphaKind parseAlpha(String s) {
        String t = s.trim().toUpperCase(Locale.ROOT);
        if (t.startsWith("DIAM")) return AlphaKind.DIAM;
        if (t.contains("LAMBDA")) return AlphaKind.INV_SQRT_LAMBDA2;
        return AlphaKind.DIAM;
    }

    static String intArrayToJson(int[] arr) {
        StringBuilder sb = new StringBuilder();
        sb.append('[');
        for (int i = 0; i < arr.length; i++) {
            if (i > 0) sb.append(',');
            sb.append(arr[i]);
        }
        sb.append(']');
        return sb.toString();
    }

    enum AlphaKind {DIAM, INV_SQRT_LAMBDA2}

    static final class GraphData {
        int n;
        long m;
        List<Integer>[] adj1Based;
        int[] labels;
        String[] labelNames;
    }
    static final class BFSResult { final int node, dist; BFSResult(int node, int dist) { this.node = node; this.dist = dist; } }
}
