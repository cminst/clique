import UsacoProbs.Triangles.Triangles
import java.io.BufferedWriter
import java.io.FileReader
import java.io.FileWriter
import java.io.PrintWriter
import java.util.*

class df {

    fun addOccurences(x: Int, y: Int, points: HashMap<Int, HashSet<Int>>) {
        if (points.containsKey(x)) {
            points[x]!!.add(y)
        } else {
            val hashset = HashSet<Int>(y)
            points[x] = hashset
        }
    }

    fun df() {

        val r = Scanner(FileReader("triangles.in"))
        val pw = PrintWriter(BufferedWriter(FileWriter("triangles.out")))
        val nPoints = r.nextInt()
        val points = HashMap<Int, HashSet<Int>>()
        for (i in 0 until nPoints) {
            val x = r.nextInt()
            val y = r.nextInt()
            addOccurences(x, y, points)
        }

        for (i in points.keys) {
            if (points[i]!!.size > 1) {
            }
        }
    }
}