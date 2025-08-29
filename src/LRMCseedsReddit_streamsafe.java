import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.lang.reflect.Method;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Consumer;
import java.util.stream.IntStream;

/**
 * LRMCseedsReddit_streamsafe.java (Optimized Version)
 *
 * Memory-lean-ish seeds exporter for Reddit with performance optimizations:
 * - Parallel graph loading using memory-mapped files
 * - Better memory allocation strategies
 * - Optimized data structures
 *
 * Usage:
 *   java -XX:+UseG1GC -XX:+UnlockExperimentalVMOptions -XX:+UseStringDeduplication \
 *        -XX:NewRatio=1 -Xmx16g -Xms8g -XX:+AlwaysPreTouch \
 *        LRMCseedsReddit_streamsafe reddit_edges.txt seeds_reddit.json [DIAM|INV_SQRT_LAMBDA2] [epsilon]
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

        System.out.println("# Loading Reddit edge list with optimized parallel loader...");
        long startTime = System.currentTimeMillis();

        GraphData G = loadRedditEdgeListOptimized(edgesPath);

        long loadTime = System.currentTimeMillis() - startTime;
        System.out.printf(Locale.US, "# Loaded Reddit edge list: n=%d, m=%d (%.2f seconds)%n",
                         G.n, G.m, loadTime / 1000.0);

        PeakTracker tracker = new PeakTracker(G, eps, alphaKind);

        System.out.println("# Found streaming entry point. Running streaming reconstruction...");
        clique2_ablations_streaming.runLaplacianRMCStreaming(G.adj1Based, tracker);

        long algoTime = System.currentTimeMillis() - startTime;
        System.out.printf(Locale.US, "# Algorithm completed in %.2f seconds%n", algoTime / 1000.0);

        tracker.writeJson(outSeeds);
        System.out.println("# Done. wrote " + outSeeds.toAbsolutePath());
    }

    // Streaming peak tracker
    static final class PeakTracker implements Consumer<clique2_ablations_streaming.SnapshotDTO> {
        final GraphData G;
        final double epsilon;
        final AlphaKind alphaKind;

        final boolean[] inC;
        final Map<Integer, Integer> bestIdxByComp = new ConcurrentHashMap<>();
        final Map<Integer, Double> bestScoreByComp = new ConcurrentHashMap<>();
        final List<Rec> arrivals = Collections.synchronizedList(new ArrayList<>());
        final AtomicInteger idx = new AtomicInteger(0);

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
        public void accept(clique2_ablations_streaming.SnapshotDTO s) {
            final int[] nodes = s.nodes;
            final int k = nodes.length;
            if (k == 0) return;

            // Mark nodes as in component (thread-safe for this use case)
            synchronized(inC) {
                for (int u : nodes) inC[u] = true;
            }

            final double dbar = s.sumDegIn / Math.max(1.0, k);
            final double Q = s.Q;
            final double alpha = (alphaKind == AlphaKind.DIAM)
                    ? approxDiameterOptimized(nodes, G.adj1Based, inC)
                    : 1.0; // simple fallback for lambda2

            final double sc = k / (Q + epsilon);
            final int compId = getSnapshotComponentId(s, nodes);
            final int sid = idx.getAndIncrement();

            bestScoreByComp.merge(compId, sc, (oldScore, newScore) -> {
                if (newScore > oldScore) {
                    bestIdxByComp.put(compId, sid);
                    return newScore;
                }
                return oldScore;
            });

            arrivals.add(new Rec(compId, sid, sc, Arrays.copyOf(nodes, nodes.length)));

            synchronized(inC) {
                for (int u : nodes) inC[u] = false;
            }
        }

        void writeJson(Path outJson) throws IOException {
            final int n = G.n;
            boolean[] covered = new boolean[n];

            try (BufferedWriter w = Files.newBufferedWriter(outJson, StandardCharsets.UTF_8)) {
                w.write("{\n");
                w.write("\"meta\":{");
                w.write("\"epsilon\":" + epsilon);
                w.write(",\"alpha_kind\":\"" + (alphaKind == AlphaKind.DIAM ? "DIAM" : "INV_SQRT_LAMBDA2") + "\"");
                w.write(",\"n\":" + G.n);
                w.write(",\"m\":" + G.m);
                w.write(",\"mode\":\"peaks_per_component+singletons(optimized-stream-or-fallback)\"");
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
                        for (int u : r.nodes) if (!covered[u]) { covered[u] = true; }
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
            }
            if (lineBuilder.length() > 0) {
                lines.add(lineBuilder.toString());
            }

            // Process lines in parallel
            lines.parallelStream()
                .filter(s -> !s.trim().isEmpty() && !s.startsWith("#"))
                .forEach(s -> {
                    String[] tok = s.split("\\s+|,");
                    if (tok.length >= 2) {
                        try {
                            int u = Integer.parseInt(tok[0]);
                            int v = Integer.parseInt(tok[1]);
                            if (u != v) {
                                degreeMap.computeIfAbsent(u, k -> new AtomicInteger(0)).incrementAndGet();
                                degreeMap.computeIfAbsent(v, k -> new AtomicInteger(0)).incrementAndGet();
                                maxNodeRef.updateAndGet(max -> Math.max(max, Math.max(u, v)));
                                if (u < v) edgeCountRef.incrementAndGet();
                            }
                        } catch (NumberFormatException e) {
                            // Skip malformed lines
                        }
                    }
                });

            return buildGraphFromData(lines, maxNodeRef.get(), edgeCountRef.get(), degreeMap);
        }
    }

    static GraphData loadWithParallelStreams(Path edgesFile) throws IOException {
        System.out.println("# Using parallel stream loading...");

        List<String> allLines = Files.readAllLines(edgesFile, StandardCharsets.UTF_8);

        // First pass: analyze in parallel
        AtomicInteger maxNodeRef = new AtomicInteger(-1);
        AtomicLong edgeCountRef = new AtomicLong(0);
        ConcurrentHashMap<Integer, AtomicInteger> degreeMap = new ConcurrentHashMap<>();

        allLines.parallelStream()
            .filter(s -> !s.trim().isEmpty() && !s.startsWith("#"))
            .forEach(s -> {
                String[] tok = s.split("\\s+|,");
                if (tok.length >= 2) {
                    try {
                        int u = Integer.parseInt(tok[0]);
                        int v = Integer.parseInt(tok[1]);
                        if (u != v) {
                            degreeMap.computeIfAbsent(u, k -> new AtomicInteger(0)).incrementAndGet();
                            degreeMap.computeIfAbsent(v, k -> new AtomicInteger(0)).incrementAndGet();
                            maxNodeRef.updateAndGet(max -> Math.max(max, Math.max(u, v)));
                            if (u < v) edgeCountRef.incrementAndGet();
                        }
                    } catch (NumberFormatException e) {
                        // Skip malformed lines
                    }
                }
            });

        return buildGraphFromData(allLines, maxNodeRef.get(), edgeCountRef.get(), degreeMap);
    }

    static GraphData buildGraphFromData(List<String> lines, int maxNode, long edgeCount,
                                       ConcurrentHashMap<Integer, AtomicInteger> degreeMap) {
        final int n = maxNode + 1;

        // Pre-allocate adjacency lists with known sizes
        @SuppressWarnings("unchecked")
        List<Integer>[] adj1 = (List<Integer>[]) new List<?>[n + 1];

        // Initialize with appropriate capacities
        IntStream.rangeClosed(1, n).parallel().forEach(i -> {
            AtomicInteger degCounter = degreeMap.get(i - 1);
            int expectedDegree = (degCounter != null) ? degCounter.get() : 0;
            // Over-allocate by 25% to reduce resizing
            int capacity = Math.max(4, (int)(expectedDegree * 1.25));
            adj1[i] = Collections.synchronizedList(new ArrayList<>(capacity));
        });

        // Second pass: build adjacency lists in parallel
        lines.parallelStream()
            .filter(s -> !s.trim().isEmpty() && !s.startsWith("#"))
            .forEach(s -> {
                String[] tok = s.split("\\s+|,");
                if (tok.length >= 2) {
                    try {
                        int u = Integer.parseInt(tok[0]);
                        int v = Integer.parseInt(tok[1]);
                        if (u != v) {
                            adj1[u + 1].add(v + 1);
                            adj1[v + 1].add(u + 1);
                        }
                    } catch (NumberFormatException e) {
                        // Skip malformed lines
                    }
                }
            });

        GraphData G = new GraphData();
        G.n = n;
        G.m = edgeCount;
        G.adj1Based = adj1;
        G.labels = new int[n];
        Arrays.fill(G.labels, -1);
        G.labelNames = new String[0];
        return G;
    }

    static double approxDiameter(int[] nodes, List<Integer>[] adj1, boolean[] inC) {
        if (nodes.length <= 1) return 0.0;

        // For larger components, sample a few starting points and take the best
        int numSamples = Math.min(3, nodes.length);
        double maxDiam = 0.0;

        for (int i = 0; i < numSamples; i++) {
            int start = nodes[i * nodes.length / numSamples];
            BFSResult a = bfsFarthest(start, adj1, inC);
            BFSResult b = bfsFarthest(a.node, adj1, inC);
            maxDiam = Math.max(maxDiam, b.dist);
        }

        return maxDiam;
    }

    static BFSResult bfsFarthest(int src, List<Integer>[] adj1, boolean[] inC) {
        int nTot = inC.length;
        int[] dist = new int[nTot];
        Arrays.fill(dist, -1);
        ArrayDeque<Integer> q = new ArrayDeque<>(1024); // Pre-size queue
        q.add(src);
        dist[src] = 0;
        int bestNode = src, bestDist = 0;

        while (!q.isEmpty()) {
            int u = q.removeFirst();
            int du = dist[u];
            if (du > bestDist) { bestDist = du; bestNode = u; }

            List<Integer> neighbors = adj1[u + 1];
            for (int i = 0, size = neighbors.size(); i < size; i++) {
                int v1 = neighbors.get(i);
                int v = v1 - 1;
                if (!inC[v] || dist[v] >= 0) continue;
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
        StringBuilder sb = new StringBuilder(arr.length * 6); // Pre-size
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
