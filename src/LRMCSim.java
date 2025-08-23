import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

public class LRMCSim {

    static final double EPSILON = 1e-6;
    static final int NUM_CLUSTERS = 10;
    static final double CLUSTER_FRACTION = 0.20;
    static final double PINTRA = 0.005;
    static final double PINTER = 0.00001;
    static final long SEED = 123456789L;

    static final int S_MIN = 10_000;
    static final int S_MAX = 1_000_000;
    static final int NUM_SIZES = 30;
    static final int TRIALS = 3;

    static final double[] DENSITY_SCALES = {0.5, 1.0, 2.0};
    static final String[] SERIES_LABELS = {"dens_x0.5", "dens_x1.0", "dens_x2.0"};

    static final boolean ECHO_CLIQUE2_OUTPUT = false;
    static final String EXTRA_HEAP = "-Xmx8g";

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

        for (int si = 0; si < DENSITY_SCALES.length; si++) {
            double scale = DENSITY_SCALES[si];
            String series = SERIES_LABELS[si];

            for (int n : sizes) {
                Path inputFile = Files.createTempFile("clique2_input_" + series + "_n" + n + "_", ".txt");
                inputFile.toFile().deleteOnExit();

                long m = generateClusteredGraphToFile(
                        n, NUM_CLUSTERS, CLUSTER_FRACTION,
                        PINTRA * scale, PINTER * scale, rng, inputFile);

                double theoX = (n + m) * Math.log(Math.max(2, n));

                for (int t = 0; t < TRIALS; t++) {
                    double ms = runClique2(EPSILON, inputFile);
                    allRows.add(new Row(series, n, m, t + 1, ms, theoX, 0));
                }
            }
        }

        double num = 0, den = 0;
        for (Row r : allRows) {
            num += r.theoX * r.ms;
            den += r.theoX * r.theoX;
        }
        double k = den == 0 ? 0 : num / den;

        // Write CSV to specified file
        try (BufferedWriter writer = Files.newBufferedWriter(Paths.get(outputCsvFile), StandardCharsets.UTF_8)) {
            writer.write("series,n,m,trial,ms,theo_x,normalized_theory_ms\n");
            for (Row r : allRows) {
                double norm = k * r.theoX;
                writer.write(String.format(Locale.US, "%s,%d,%d,%d,%.3f,%.3f,%.3f\n",
                        r.series, r.n, r.m, r.trial, r.ms, r.theoX, norm));
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
                double norm = k * theoX;
                System.out.printf(Locale.US, "# summary,%s,%d,%d,%.3f,%.3f,%.3f%n",
                        s, n, m, mean, theoX, norm);
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
        cmd.add(Double.toString(epsilon));
        cmd.add(inputFile.toAbsolutePath().toString());

        ProcessBuilder pb = new ProcessBuilder(cmd);
        pb.redirectErrorStream(true);
        Process p = pb.start();

        String lastRuntime = null;
        try (BufferedReader br = new BufferedReader(
                new InputStreamReader(p.getInputStream(), StandardCharsets.UTF_8))) {
            String line;
            while ((line = br.readLine()) != null) {
                if (ECHO_CLIQUE2_OUTPUT) System.out.println(line);
                if (line.startsWith("Runtime:")) lastRuntime = line;
            }
        }
        int exit = p.waitFor();
        if (exit != 0) throw new RuntimeException("clique2 exited with code " + exit);
        if (lastRuntime == null) throw new RuntimeException("No 'Runtime: ... ms' line from clique2");

        String msStr = lastRuntime.replace("Runtime:", "").replace("ms", "").trim();
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
            for (int i = 0; i < k; i++) m += triPairsToWriter(clusters[i], pIntra, w, rng);
            for (int i = 0; i < k; i++) for (int j = i + 1; j < k; j++)
                m += rectPairsToWriter(clusters[i], clusters[j], pInter, w, rng);
            for (int i = 0; i < k; i++) m += rectPairsToWriter(clusters[i], background, pInter, w, rng);
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
        final String series; final int n; final long m; final int trial; final double ms; final double theoX; final double norm;
        Row(String series, int n, long m, int trial, double ms, double theoX, double norm) {
            this.series = series; this.n = n; this.m = m; this.trial = trial; this.ms = ms; this.theoX = theoX; this.norm = norm;
        }
    }
}
