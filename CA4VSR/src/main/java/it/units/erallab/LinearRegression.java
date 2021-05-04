package it.units.erallab;

import it.units.erallab.hmsrobots.util.Point2;
import java.util.List;

// https://algs4.cs.princeton.edu/code/edu/princeton/cs/algs4/LinearRegression.java.html

public class LinearRegression {
    private final double intercept, slope;
    private final double r2;
    private final double svar0, svar1;
    private final double residualSumSquared;

    public LinearRegression(List<Point2> data) {
        int n = data.size();
        if (n < 2) {
            this.r2 = 0;
            this.slope = 0;
            this.intercept = 0;
            this.residualSumSquared = 0;
            this.svar0 = 0;
            this.svar1 = 0;
        } else {
            // first pass
            double sumx = 0.0, sumy = 0.0, sumx2 = 0.0;
            for (Point2 point : data) {
                sumx  += point.x;
                sumx2 += point.x*point.x;
                sumy  += point.y;
            }
            double xbar = sumx / n;
            double ybar = sumy / n;

            // second pass: compute summary statistics
            double xxbar = 0.0, yybar = 0.0, xybar = 0.0;
            for (Point2 point : data) {
                xxbar += (point.x - xbar) * (point.x - xbar);
                yybar += (point.y - ybar) * (point.y - ybar);
                xybar += (point.x - xbar) * (point.y - ybar);
            }

            slope  = xybar / xxbar;

            intercept = ybar - slope * xbar;

            // more statistical analysis
            double rss = 0.0;      // residual sum of squares
            double ssr = 0.0;      // regression sum of squares
            for (Point2 point : data) {
                double fit = slope*point.x + intercept;
                rss += (fit - point.y) * (fit - point.y);
                ssr += (fit - ybar) * (fit - ybar);
            }
            residualSumSquared = rss;

            int degreesOfFreedom = n-2;
            r2    = ssr / yybar;
            double svar  = rss / degreesOfFreedom;
            svar1 = svar / xxbar;
            svar0 = svar/n + xbar*xbar*svar1;
        }
    }

    public double intercept() {
        return intercept;
    }

    public double slope() {
        return slope;
    }

    public double R2() {
        return r2;
    }

    public double predict(double x) {
        return slope*x + intercept;
    }

    public String toString() {
        StringBuilder s = new StringBuilder();
        s.append(String.format("%.2f n + %.2f", slope(), intercept()));
        s.append("  (R^2 = " + String.format("%.3f", R2()) + ")");
        return s.toString();
    }
}
