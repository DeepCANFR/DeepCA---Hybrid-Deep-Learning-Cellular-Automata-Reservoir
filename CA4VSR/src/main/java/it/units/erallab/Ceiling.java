package it.units.erallab;

import it.units.erallab.hmsrobots.core.objects.WorldObject;
import it.units.erallab.hmsrobots.core.objects.immutable.Immutable;
import it.units.erallab.hmsrobots.util.Point2;
import it.units.erallab.hmsrobots.util.Poly;
import org.dyn4j.dynamics.Body;
import org.dyn4j.dynamics.World;
import org.dyn4j.geometry.MassType;
import org.dyn4j.geometry.Polygon;
import org.dyn4j.geometry.Vector2;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Ceiling implements WorldObject {
    private static final double MIN_Y_THICKNESS = 10d;

    private final double[] xs;
    private final double[] ys;
    private final List<Body> bodies;
    private final List<Vector2> polygon;

    public Ceiling(double[] xs, double[] ys) {
        this.xs = xs;
        this.ys = ys;
        if (xs.length != ys.length) {
            throw new IllegalArgumentException("xs[] and ys[] must have the same length");
        }
        if (xs.length < 2) {
            throw new IllegalArgumentException("There must be at least 2 points");
        }
        double[] sortedXs = Arrays.copyOf(xs, xs.length);
        Arrays.sort(sortedXs);
        if (!Arrays.equals(xs, sortedXs)) {
            throw new IllegalArgumentException("x coordinates must be sorted");
        }
        //init collections
        bodies = new ArrayList<>(xs.length - 1);
        polygon = new ArrayList<>(xs.length + 2);
        //find min y
        double baseY = Arrays.stream(ys).max().getAsDouble()+MIN_Y_THICKNESS;
        polygon.add(new Vector2(0, baseY));
        //build bodies and polygon
        for (int i = 1; i < xs.length; i++) {
            Polygon bodyPoly = new Polygon(
                    new Vector2(0, baseY),
                    new Vector2(0, ys[i - 1]),
                    new Vector2(xs[i] - xs[i - 1], ys[i]),
                    new Vector2(xs[i] - xs[i - 1], baseY)
            );
            Body body = new Body(1);
            body.addFixture(bodyPoly);
            body.setMass(MassType.INFINITE);
            body.translate(xs[i - 1], 0);
            //body.translate(-xs[0], -minY);
            bodies.add(body);
            polygon.add(new Vector2(xs[i - 1], ys[i - 1]));
        }
        polygon.add(new Vector2(xs[xs.length - 1], ys[xs.length - 1]));
        polygon.add(new Vector2(xs[xs.length - 1], baseY));
    }

    @Override
    public Immutable immutable() {
        Point2[] vertices = new Point2[polygon.size()];
        for (int i = 0; i < vertices.length; i++) {
            vertices[i] = Point2.build(polygon.get(i));
        }
        return new it.units.erallab.hmsrobots.core.objects.immutable.Ground(Poly.build(vertices));
    }

    @Override
    public void addTo(World world) {
        for (Body body : bodies) {
            world.addBody(body);
        }
    }

    public List<Body> getBodies() {
        return bodies;
    }

    public double yAt(double x) {
        double y = Double.NEGATIVE_INFINITY;
        for (int i = 1; i < xs.length; i++) {
            if ((xs[i - 1] <= x) && (x <= xs[i])) {
                return (x - xs[i - 1]) * (ys[i] - ys[i - 1]) / (xs[i] - xs[i - 1]) + ys[i - 1];
            }
        }
        return Double.NaN;
    }
}
