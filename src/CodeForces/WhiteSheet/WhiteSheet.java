package CodeForces.WhiteSheet;

import java.awt.*;
import java.util.HashSet;
import java.util.Scanner;
import java.util.Vector;

public class WhiteSheet {

    public static void main(String[] args) {
        Scanner r = new Scanner(System.in);
        Point[] points = new Point[6];

        for (int i = 0; i < 6; i++) {
            int x = r.nextInt();
            int y = r.nextInt();
            points[i] = new Point(x, y);
        }

        Rectangle white = new Rectangle(points[0].x, points[0].y, points[1].x - points[0].x, points[1].y - points[0].y);
        Rectangle black1 = new Rectangle(points[2].x, points[2].y, points[3].x - points[2].x, points[3].y - points[2].y);
        Rectangle black2 = new Rectangle(points[4].x, points[4].y, points[5].x - points[4].x, points[5].y - points[4].y);

        Point[] whitePoints = new Point[]{new Point(white.x, white.y), new Point(white.x + white.width, white.y), new Point(white.x, white.y + white.height), new Point(white.x + white.width, white.y + white.height)};

        boolean containsAll1 = true;
        boolean containsAll2 = true;
        for (Point i : whitePoints) {
            if (!containsAll1 && !containsAll2) break;
            if (!containsPoint(black1, i)) containsAll1 = false;
            if (!containsPoint(black2, i)) containsAll2 = false;
        }
        if (containsAll1 || containsAll2) {
            System.out.println("NO");
            return;
        } else if (cornersCovered(black1, white) == 2 && cornersCovered(black2, white) == 2) {
            //            check Corners covered for the 2 black sheets
            Vector<Corner> black1Corners = cornersCoveredExactly(black1, white);
            Vector<Corner> black2Corners = cornersCoveredExactly(black2, white);

            HashSet<Corner> uniqueCorners = new HashSet<>(black1Corners);
            uniqueCorners.addAll(black2Corners);
            if (uniqueCorners.size() == 4) {
                if (checkOverlap(black1, black2)) {
                    System.out.println("NO");
                    return;
                }
            }
        }
        System.out.println("YES");
    }

    public static int cornersCovered(Rectangle black, Rectangle white) { //return how many corners black count on white
        int count = 0;
        Point lowerRight = new Point(white.x + white.width, white.y);
        Point upperRight = new Point(white.x + white.width, white.y + white.height);
        Point upperLeft = new Point(white.x, white.y + white.height);

        if ((white.x <= (black.x + black.width) && white.x >= black.x) && (white.y >= black.y && white.y <= black.y + black.height))
            count++;
        if (containsPoint(black, lowerRight)) count++;
        if (containsPoint(black, upperRight)) count++;
        if (containsPoint(black, upperLeft)) count++;

        return count;
    }

    public static Vector<Corner> cornersCoveredExactly(Rectangle black, Rectangle white) { // what was covered exactly?
        Vector<Corner> corners = new Vector<>();
        int lowerXWhite = white.x;
        int lowerYWhite = white.y;
        int higherXWhite = white.x + white.width;
        int higherYWhite = white.y + white.height;
        if (containsPoint(black, new Point(higherXWhite, higherYWhite))) {
            corners.add(Corner.UPPER_RIGHT);
        }
        if (containsPoint(black, new Point(higherXWhite, lowerYWhite))) {
            corners.add(Corner.LOWER_RIGHT);
        }
        if (containsPoint(black, new Point(lowerXWhite, higherYWhite))) {
            corners.add(Corner.UPPER_LEFT);
        }
        if (containsPoint(black, new Point(lowerXWhite, lowerYWhite))) {
            corners.add(Corner.LOWER_LEFT);
        }
        return corners;
    }

    static boolean checkOverlap(Rectangle rect1, Rectangle rect2) { // return if 2 rects overlap
        int minX = Math.min(rect1.x, rect2.x);
        int minY = Math.min(rect1.y, rect2.y);
        int maxX = Math.max(rect1.x + rect1.width, rect2.x + rect2.width);
        int maxY = Math.max(rect1.y + rect1.height, rect2.y + rect2.height);
        return (maxY - minY <= rect1.height + rect2.height) && (maxX - minX <= rect1.width + rect2.width);
    }

    public static boolean containsPoint(Rectangle rect, Point point) { // does rect contain point?
        return (point.x <= rect.x + rect.width && point.x >= rect.x) && (point.y <= rect.y + rect.height && point.y >= rect.y);
    }

    enum Corner {
        LOWER_LEFT, LOWER_RIGHT, UPPER_LEFT, UPPER_RIGHT
    }
}