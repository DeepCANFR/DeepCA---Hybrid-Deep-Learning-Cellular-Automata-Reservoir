package it.units.erallab;

import it.units.erallab.hmsrobots.core.objects.ControllableVoxel;
import it.units.erallab.hmsrobots.core.objects.Ground;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.WorldObject;
import it.units.erallab.hmsrobots.core.objects.immutable.Snapshot;
import it.units.erallab.hmsrobots.core.objects.immutable.Voxel;
import it.units.erallab.hmsrobots.tasks.AbstractTask;
import it.units.erallab.hmsrobots.util.BoundingBox;
import it.units.erallab.hmsrobots.util.Point2;
import it.units.erallab.hmsrobots.viewers.SnapshotListener;
import org.dyn4j.dynamics.Settings;
import org.dyn4j.dynamics.World;
import org.dyn4j.geometry.Vector2;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

public class Escape extends AbstractTask<Robot<?>, List<Double>> {

    private final static double INITIAL_PLACEMENT_X_GAP = 1d;
    private final static double INITIAL_PLACEMENT_Y_GAP = 1d;
    private final static double TERRAIN_BORDER_HEIGHT = 100d;
    public static final int TERRAIN_LENGHT = 2000;

    public enum Metric {
        TRAVELED_X_DISTANCE,
        CENTER_MAX_Y,
        TRAVEL_X_VELOCITY,
        TRAVEL_X_RELATIVE_VELOCITY,
        CENTER_AVG_Y,
        CONTROL_POWER,
        RELATIVE_CONTROL_POWER,
        AREA_RATIO_POWER,
        RELATIVE_AREA_RATIO_POWER,
        X_DISTANCE_CORRECTED_EFFICIENCY;
    }

    private final double finalT;
    private final double[][] groundProfile;
    private final double initialPlacement;
    private final List<it.units.erallab.hmsrobots.tasks.Locomotion.Metric> metrics;

    public Escape(double finalT, List<it.units.erallab.hmsrobots.tasks.Locomotion.Metric> metrics, Settings settings) {
        this(finalT, new double[][]{{0, 1, 200},{100, 0, 0}}, new double[][]{{0, 1, 200},{100, 0, 0}}[0][1] + INITIAL_PLACEMENT_X_GAP, metrics, settings);
    }

    public Escape(double finalT, double[][] groundProfile, double initialPlacement, List<it.units.erallab.hmsrobots.tasks.Locomotion.Metric> metrics, Settings settings) {
        super(settings);
        this.finalT = finalT;
        this.groundProfile = groundProfile;
        this.initialPlacement = initialPlacement;
        this.metrics = metrics;
    }

    @Override
    public List<Double> apply(Robot<?> robot, SnapshotListener listener) {
        List<Point2> centerPositions = new ArrayList<>();
        //init world
        World world = new World();
        world.setSettings(settings);
        List<WorldObject> worldObjects = new ArrayList<>();

        Ground ground = new Ground(new double[]{0, 1, 200}, new double[]{100, 0, 0});

        double maxY = robot.boundingBox().max.y;
        double maxX = robot.boundingBox().max.x;

        Ceiling ceiling = new Ceiling(new double[]{0, maxX*1.1, maxX*1.4, maxX*2.5, maxX*10}, new double[]{1.4*maxY, 1.4*maxY, maxY, 0.7*maxY, 0.6*maxY});

        ceiling.addTo(world);
        worldObjects.add(ceiling);

        ground.addTo(world);
        worldObjects.add(ground);
        //position robot: translate on x
        BoundingBox boundingBox = robot.boundingBox();
        robot.translate(new Vector2(initialPlacement - boundingBox.min.x, 0));
        //translate on y
        double minYGap = robot.getVoxels().values().stream()
                .filter(Objects::nonNull)
                .mapToDouble(v -> ((Voxel) v.immutable()).getShape().boundingBox().min.y - ground.yAt(v.getCenter().x))
                .min().orElse(0d);
        robot.translate(new Vector2(0, INITIAL_PLACEMENT_Y_GAP - minYGap));
        //get initial x
        double initCenterX = robot.getCenter().x;
        //add robot to world
        robot.addTo(world);
        worldObjects.add(robot);
        //run
        double t = 0d;
        while (t < finalT) {
            t = t + settings.getStepFrequency();
            world.step(1);
            robot.act(t);
            //update center position metrics
            centerPositions.add(Point2.build(robot.getCenter()));
            //possibly output snapshot
            if (listener != null) {
                Snapshot snapshot = new Snapshot(t, worldObjects.stream().map(WorldObject::immutable).collect(Collectors.toList()));
                listener.listen(snapshot);
            }
        }
        //compute metrics
        List<Double> results = new ArrayList<>(metrics.size());
        for (it.units.erallab.hmsrobots.tasks.Locomotion.Metric metric : metrics) {
            double value = switch (metric) {
                case TRAVELED_X_DISTANCE -> (robot.getCenter().x - initCenterX);
                case TRAVEL_X_VELOCITY -> (robot.getCenter().x - initCenterX) / t;
                case TRAVEL_X_RELATIVE_VELOCITY -> (robot.getCenter().x - initCenterX) / t / Math.max(boundingBox.max.x - boundingBox.min.x, boundingBox.max.y - boundingBox.min.y);
                case CENTER_MAX_Y -> centerPositions.stream()
                        .mapToDouble((p) -> p.y)
                        .max()
                        .orElse(0);
                case CENTER_AVG_Y -> centerPositions.stream()
                        .mapToDouble((p) -> p.y)
                        .average()
                        .orElse(0);
                case CONTROL_POWER -> robot.getVoxels().values().stream()
                        .filter(v -> (v instanceof ControllableVoxel))
                        .mapToDouble(ControllableVoxel::getControlEnergy)
                        .sum() / t;
                case RELATIVE_CONTROL_POWER -> robot.getVoxels().values().stream()
                        .filter(v -> (v instanceof ControllableVoxel))
                        .mapToDouble(ControllableVoxel::getControlEnergy)
                        .sum() / t / robot.getVoxels().values().stream().filter(Objects::nonNull).count();
                case AREA_RATIO_POWER -> robot.getVoxels().values().stream()
                        .filter(v -> (v instanceof ControllableVoxel))
                        .mapToDouble(ControllableVoxel::getAreaRatioEnergy)
                        .sum() / t;
                case RELATIVE_AREA_RATIO_POWER -> robot.getVoxels().values().stream()
                        .filter(v -> (v instanceof ControllableVoxel))
                        .mapToDouble(ControllableVoxel::getAreaRatioEnergy)
                        .sum() / t / robot.getVoxels().values().stream().filter(Objects::nonNull).count();
            };
            results.add(value);
        }
        return results;
    }

    private static double[][] randomTerrain(int n, double length, double peak, double borderHeight, Random random) {
        double[] xs = new double[n + 2];
        double[] ys = new double[n + 2];
        xs[0] = 0d;
        xs[n + 1] = length;
        ys[0] = borderHeight;
        ys[n + 1] = borderHeight;
        for (int i = 1; i < n + 1; i++) {
            xs[i] = 1 + (double) (i - 1) * (length - 2d) / (double) n;
            ys[i] = random.nextDouble() * peak;
        }
        return new double[][]{xs, ys};
    }

    public static double[][] createTerrain(String name) {
        String flat = "flat";
        String hilly = "hilly-(?<h>[0-9]+(\\.[0-9]+)?)-(?<w>[0-9]+(\\.[0-9]+)?)-(?<seed>[0-9]+)";
        String steppy = "steppy-(?<h>[0-9]+(\\.[0-9]+)?)-(?<w>[0-9]+(\\.[0-9]+)?)-(?<seed>[0-9]+)";
        if (name.matches(flat)) {
            return new double[][]{
                    new double[]{0, 10, TERRAIN_LENGHT - 10, TERRAIN_LENGHT},
                    new double[]{TERRAIN_BORDER_HEIGHT, 5, 5, TERRAIN_BORDER_HEIGHT}
            };
        }
        if (name.matches(hilly)) {
            double h = Double.parseDouble(paramValue(hilly, name, "h"));
            double w = Double.parseDouble(paramValue(hilly, name, "w"));
            Random random = new Random(Integer.parseInt(paramValue(hilly, name, "seed")));
            List<Double> xs = new ArrayList<>(List.of(0d, 10d));
            List<Double> ys = new ArrayList<>(List.of(TERRAIN_BORDER_HEIGHT, 0d));
            while (xs.get(xs.size() - 1) < TERRAIN_LENGHT) {
                xs.add(xs.get(xs.size() - 1) + Math.max(1d, (random.nextGaussian() * 0.25 + 1) * w));
                ys.add(ys.get(ys.size() - 1) + random.nextGaussian() * h);
            }
            xs.addAll(List.of(xs.get(xs.size() - 1) + 10, xs.get(xs.size() - 1) + 20));
            ys.addAll(List.of(0d, TERRAIN_BORDER_HEIGHT));
            return new double[][]{
                    xs.stream().mapToDouble(d -> d).toArray(),
                    ys.stream().mapToDouble(d -> d).toArray()
            };
        }
        if (name.matches(steppy)) {
            double h = Double.parseDouble(paramValue(steppy, name, "h"));
            double w = Double.parseDouble(paramValue(steppy, name, "w"));
            Random random = new Random(Integer.parseInt(paramValue(steppy, name, "seed")));
            List<Double> xs = new ArrayList<>(List.of(0d, 10d));
            List<Double> ys = new ArrayList<>(List.of(TERRAIN_BORDER_HEIGHT, 0d));
            while (xs.get(xs.size() - 1) < TERRAIN_LENGHT) {
                xs.add(xs.get(xs.size() - 1) + Math.max(1d, (random.nextGaussian() * 0.25 + 1) * w));
                xs.add(xs.get(xs.size() - 1) + 0.5d);
                ys.add(ys.get(ys.size() - 1));
                ys.add(ys.get(ys.size() - 1) + random.nextGaussian() * h);
            }
            xs.addAll(List.of(xs.get(xs.size() - 1) + 10, xs.get(xs.size() - 1) + 20));
            ys.addAll(List.of(0d, TERRAIN_BORDER_HEIGHT));
            return new double[][]{
                    xs.stream().mapToDouble(d -> d).toArray(),
                    ys.stream().mapToDouble(d -> d).toArray()
            };
        }
        throw new IllegalArgumentException(String.format("Unknown terrain name: %s", name));
    }

    public List<it.units.erallab.hmsrobots.tasks.Locomotion.Metric> getMetrics() {
        return metrics;
    }

    private static String paramValue(String pattern, String string, String paramName) {
        Matcher matcher = Pattern.compile(pattern).matcher(string);
        if (matcher.matches()) {
            return matcher.group(paramName);
        }
        throw new IllegalStateException(String.format("Param %s not found in %s with pattern %s", paramName, string, pattern));
    }

}

