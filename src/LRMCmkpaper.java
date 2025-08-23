import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

class LRMCmkpaper {

    // ---------------- experiment constants from the paper ----------------
    // n in [20k, 1M], 10 planted clusters, p_intra in [0.005, 0.02], q in [1e-4, 1e-3]
    static final int NUM_CLUSTERS = 10;
    static final double CLUSTER_FRACTION = 0.20; // 20% of nodes participate in planted clusters

    // Use three intra densities across the paper's range, with a fixed sparse q
    // Main runtime figure can use the middle setting: p_intra = 0.01, q = 1e-4
    static final double[] PINTRA_SERIES = {0.010};
    static final double PINTER_FIXED = 1e-4; // keep inter-links sparse and tractable at n up to 1M
    static final String[] SERIES_LABELS = {"p0.010"};

    // Trials and sizes
    static final int S_MIN = 894_000;  // paper starts at 20k
    static final int S_MAX = 1_050_000;
    static final int NUM_SIZES = 2;
    static final int TRIALS = 3;

    // Algorithm hyperparameters
    static final double EPSILON = 1e-6; // paper default

    // JVM settings for the inner run; bump if you hit OOM at the top end
    static final String EXTRA_HEAP = "-Xmx8g";

    // If your clique2 main expects epsilon before the input path, set this true
    static final boolean PASS_EPSILON = true;

    static final long SEED = 123456789L;

    static String clique2Main;
    static String outputCsvFile;

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Usage: java LRMCSim <CLIQUE2_MAIN> <output_csv_file>");
            return;
        }

        clique2Main = args[0];
        outputCsvFile = args[1];

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

                // Compute degeneracy k on the generated graph
                int kDeg = computeDegeneracyFromFile(n, m, inputFile);

                // Theory curve: (|V|+|E|) log |V| + |E| * k
                double theoX = (n + m) * Math.log(Math.max(2, n)) + (double) m * kDeg;

                for (int t = 0; t < TRIALS; t++) {
                    double ms = runClique2(EPSILON, inputFile);
                    allRows.add(new Row(series, n, m, t + 1, ms, theoX, pIntra, pInter));
                }
            }
        }

        // Fit scale so that scale * theoX ~ ms in least squares sense
        double num = 0, den = 0;
        for (Row r : allRows) {
            num += r.theoX * r.ms;
            den += r.theoX * r.theoX;
        }
        double scale = den == 0 ? 0 : num / den;

        // Write CSV to specified file
        try (BufferedWriter writer = Files.newBufferedWriter(Paths.get(outputCsvFile), StandardCharsets.UTF_8)) {
            writer.write("series,n,m,trial,ms,theo_x,normalized_theory_ms,p_intra,p_inter\n");
            for (Row r : allRows) {
                double norm = scale * r.theoX;
                writer.write(String.format(Locale.US, "%s,%d,%d,%d,%.3f,%.3f,%.3f,%.6f,%.6g\n",
                        r.series, r.n, r.m, r.trial, r.ms, r.theoX, norm, r.pIntra, r.pInter));
            }
        }

        // Optional: also echo summaries to stdout
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
        cmd.add(javaBin);
        cmd.add(EXTRA_HEAP);
        cmd.add("-cp");
        cmd.add(classpath);
        cmd.add(clique2Main);
        if (PASS_EPSILON) cmd.add(Double.toString(epsilon));
        cmd.add(inputFile.toAbsolutePath().toString());

        ProcessBuilder pb = new ProcessBuilder(cmd);
        pb.redirectErrorStream(true);
        Process p = pb.start();

        String lastRuntime = null;
        try (BufferedReader br = new BufferedReader(
                new InputStreamReader(p.getInputStream(), StandardCharsets.UTF_8))) {
            String line;
            while ((line = br.readLine()) != null) {
                if (line.startsWith("Runtime:")) lastRuntime = line;
            }
        }
        int exit = p.waitFor();
        if (exit != 0) throw new RuntimeException("clique2 exited with code " + exit);
        if (lastRuntime == null) throw new RuntimeException("No 'Runtime: ... ms' line from clique2");

        String msStr = lastRuntime.replace("Runtime:", "").replace("ms", "").trim();
        System.out.println(inputFile.toAbsolutePath().toString());
        return Double.parseDouble(msStr);
    }

    // ------------ clustered generator (expected O(m)) ------------
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
            // intra-cluster edges
            for (int i = 0; i < k; i++) m += triPairsToWriter(clusters[i], pIntra, w, rng);
            // inter-links between clusters
            for (int i = 0; i < k; i++) for (int j = i + 1; j < k; j++)
                m += rectPairsToWriter(clusters[i], clusters[j], pInter, w, rng);
            // links between clusters and background
            for (int i = 0; i < k; i++) m += rectPairsToWriter(clusters[i], background, pInter, w, rng);
            // sparse background background links as well
            m += triPairsToWriter(background, pInter, w, rng);
        }

        try (BufferedWriter hdr = Files.newBufferedWriter(outFile, StandardCharsets.UTF_8,
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)) {
            hdr.write(n + " " + m);
            hdr.newLine();
        }
        try (OutputStream out = Files.newOutputStream(outFile, StandardOpenOption.APPEND);
             InputStream in = Files.newInputStream(tmpEdges)) {
            byte[] buf = new byte[1 << 20];
            int len;
            while ((len = in.read(buf)) != -1) out.write(buf, 0, len);
        }
        return m;
    }

    // Compute graph degeneracy (k-core number) in O(n + m) time.
    // Reads the edge list from 'edgeListFile', which is the same file with the "n m" header then undirected edges.
    // Nodes are 1-based in the file and converted to 0-based internally.
    static int computeDegeneracyFromFile(int n, long m, Path edgeListFile) throws IOException {
        // 1) First pass: degrees
        int[] deg = new int[n];
        try (BufferedReader br = Files.newBufferedReader(edgeListFile, StandardCharsets.UTF_8)) {
            br.readLine(); // skip header "n m"
            String s;
            while ((s = br.readLine()) != null) {
                if (s.isEmpty()) continue;
                int sp = s.indexOf(' ');
                if (sp <= 0) continue;
                int u = Integer.parseInt(s.substring(0, sp)) - 1;
                int v = Integer.parseInt(s.substring(sp + 1)) - 1;
                if (u == v) continue; // ignore self-loops if any
                if (u < 0 || u >= n || v < 0 || v >= n) continue; // guard
                deg[u]++; deg[v]++;
            }
        }

        int maxDeg = 0;
        long totalAdj = 0;
        for (int d : deg) { if (d > maxDeg) maxDeg = d; totalAdj += d; }

        // 2) Build adjacency with offsets, second pass to fill
        if (totalAdj > Integer.MAX_VALUE) {
            // Fallback: return an upper bound when the adjacency would overflow.
            int ub = 0; for (int d : deg) if (d > ub) ub = d;
            return ub;
        }
        int[] off = new int[n + 1];
        for (int i = 0; i < n; i++) off[i + 1] = off[i] + deg[i];
        int[] adj = new int[(int) totalAdj];
        int[] cur = Arrays.copyOf(off, off.length);

        try (BufferedReader br = Files.newBufferedReader(edgeListFile, StandardCharsets.UTF_8)) {
            br.readLine(); // skip header
            String s;
            while ((s = br.readLine()) != null) {
                if (s.isEmpty()) continue;
                int sp = s.indexOf(' ');
                if (sp <= 0) continue;
                int u = Integer.parseInt(s.substring(0, sp)) - 1;
                int v = Integer.parseInt(s.substring(sp + 1)) - 1;
                if (u == v || u < 0 || u >= n || v < 0 || v >= n) continue;
                adj[cur[u]++] = v;
                adj[cur[v]++] = u;
            }
        }

        // 3) Matula–Beck bucket-based core decomposition
        int[] degree = Arrays.copyOf(deg, deg.length);
        int[] bin = new int[maxDeg + 1];
        for (int d : degree) bin[d]++;

        int start = 0;
        for (int d = 0; d <= maxDeg; d++) {
            int count = bin[d];
            bin[d] = start;
            start += count;
        }

        int[] vert = new int[n];
        int[] pos  = new int[n];
        for (int v = 0; v < n; v++) {
            pos[v] = bin[degree[v]];
            vert[pos[v]] = v;
            bin[degree[v]]++;
        }
        // reset bin to starts
        for (int d = maxDeg; d > 0; d--) bin[d] = bin[d - 1];
        bin[0] = 0;

        int kDeg = 0;
        for (int i = 0; i < n; i++) {
            int v = vert[i];
            int dv = degree[v];
            if (dv > kDeg) kDeg = dv;

            // peel v and decrement neighbors with degree > dv
            for (int p = off[v]; p < off[v + 1]; p++) {
                int u = adj[p];
                if (degree[u] > dv) {
                    int du = degree[u];
                    int pu = pos[u];
                    int pw = bin[du];
                    int w  = vert[pw];
                    if (u != w) {
                        // swap u with w
                        vert[pu] = w; pos[w] = pu;
                        vert[pw] = u; pos[u] = pw;
                    }
                    bin[du]++;         // advance bucket start for degree du
                    degree[u] = du - 1;
                }
            }
            degree[v] = 0; // removed
        }
        return kDeg;
    }

    // skip-sampling over unordered pairs in a set
    static long triPairsToWriter(int[] set, double p, BufferedWriter w, Random rng) throws IOException {
        int s = set.length; if (s < 2 || p <= 0) return 0L;
        final double logq = Math.log(1.0 - p);
        long written = 0;
        int row = 0, off = -1;
        while (row < s - 1) {
            double r = rng.nextDouble();
            int skip = (int) Math.floor(Math.log(1.0 - r) / logq);
            off += 1 + skip;
            while (row < s - 1 && off >= (s - row - 1)) {
                off -= (s - row - 1);
                row++;
            }
            if (row < s - 1) {
                int u = set[row], v = set[row + 1 + off];
                w.write(u + " " + v); w.newLine();
                written++;
            }
        }
        return written;
    }

    // skip-sampling over A x B
    static long rectPairsToWriter(int[] A, int[] B, double p, BufferedWriter w, Random rng) throws IOException {
        int a = A.length, b = B.length; if (a == 0 || b == 0 || p <= 0) return 0L;
        final double logq = Math.log(1.0 - p);
        long total = 1L * a * b, t = -1, written = 0;
        while (true) {
            double r = rng.nextDouble();
            long skip = (long) Math.floor(Math.log(1.0 - r) / logq);
            t += 1 + skip;
            if (t >= total) break;
            int i = (int) (t / b), j = (int) (t % b);
            w.write(A[i] + " " + B[j]); w.newLine();
            written++;
        }
        return written;
    }

    // ---------------- helpers ----------------
    static int[] logSpaced(int lo, int hi, int k) {
        double a = Math.log(lo), b = Math.log(hi);
        int[] out = new int[k];
        for (int i = 0; i < k; i++) {
            double t = i / (double) (k - 1);
            out[i] = (int) Math.round(Math.exp(a + t * (b - a)));
            out[i] = Math.max(lo, Math.min(hi, (out[i] + 500) / 1000 * 1000)); // snap to nearest 1000
        }
        for (int i = 1; i < k; i++) if (out[i] <= out[i - 1]) out[i] = out[i - 1] + 1000;
        out[k - 1] = hi;
        return out;
    }

    static void shuffle(int[] a, Random rng) {
        for (int i = a.length - 1; i > 0; i--) { int j = rng.nextInt(i + 1); int t = a[i]; a[i] = a[j]; a[j] = t; }
    }

    static double mean(double[] x) { double s = 0; for (double v : x) s += v; return s / x.length; }
    static double stddev(double[] x, double mean) {
        if (x.length <= 1) return 0;
        double s2 = 0; for (double v : x) { double d = v - mean; s2 += d * d; }
        return Math.sqrt(s2 / (x.length - 1));
    }

    static final class Row {
        final String series; final int n; final long m; final int trial; final double ms; final double theoX; final double pIntra; final double pInter;
        Row(String series, int n, long m, int trial, double ms, double theoX, double pIntra, double pInter) {
            this.series = series; this.n = n; this.m = m; this.trial = trial; this.ms = ms; this.theoX = theoX; this.pIntra = pIntra; this.pInter = pInter;
        }
    }
}
