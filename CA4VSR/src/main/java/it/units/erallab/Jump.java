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
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.stream.Collectors;
import org.dyn4j.dynamics.Settings;
import org.dyn4j.dynamics.World;
import org.dyn4j.geometry.Vector2;

public class Jump extends AbstractTask<Robot<?>, List<Double>> {
    private static final double INITIAL_PLACEMENT_X_GAP = 1.0D;
    private static final double INITIAL_PLACEMENT_Y_GAP = 1.0D;
    private static final double TERRAIN_BORDER_HEIGHT = 100.0D;
    private static final int TERRAIN_POINTS = 50;
    private final double finalT;
    private final double[][] groundProfile;
    private final double initialPlacement;
    private final List<it.units.erallab.Jump.Metric> metrics;

    public Jump(double finalT, double[][] groundProfile, List<it.units.erallab.Jump.Metric> metrics, Settings settings) {
        this(finalT, groundProfile, groundProfile[0][1] + 1.0D, metrics, settings);
    }

    public Jump(double finalT, double[][] groundProfile, double initialPlacement, List<it.units.erallab.Jump.Metric> metrics, Settings settings) {
        super(settings);
        this.finalT = finalT;
        this.groundProfile = groundProfile;
        this.initialPlacement = initialPlacement;
        this.metrics = metrics;
    }

    public List<Double> apply(Robot<?> robot, SnapshotListener listener) {
        List<Point2> centerPositions = new ArrayList();
        World world = new World();
        world.setSettings(this.settings);
        List<WorldObject> worldObjects = new ArrayList();
        Ground ground = new Ground(this.groundProfile[0], this.groundProfile[1]);
        ground.addTo(world);
        worldObjects.add(ground);
        BoundingBox boundingBox = robot.boundingBox();
        robot.translate(new Vector2(this.initialPlacement - boundingBox.min.x, 0.0D));
        double minYGap = robot.getVoxels().values().stream().filter(Objects::nonNull).mapToDouble((v) -> {
            return ((Voxel)v.immutable()).getShape().boundingBox().min.y - ground.yAt(v.getCenter().x);
        }).min().orElse(0.0D);
        robot.translate(new Vector2(0.0D, 1.0D - minYGap));
        robot.addTo(world);
        worldObjects.add(robot);

        // wait for 10 secs before simulation
        double transitory = 10.0;
        double transitoryStep = 0.0D;
        while (transitoryStep < transitory) {
            transitoryStep += this.settings.getStepFrequency();
            world.step(1);
        }

        double maxYTime = 0.0;
        double maxY = 0.0;
        double controlEnergy = 0.0;

        double t = 0.0D;
        while(t < this.finalT) {
            t += this.settings.getStepFrequency();
            world.step(1);
            robot.act(t);
            if (robot.getCenter().y > maxY) {
                maxY = robot.getCenter().y;
                maxYTime = t;

                controlEnergy = robot.getVoxels().values().stream().filter((v) -> {
                    return v instanceof ControllableVoxel;
                }).mapToDouble(ControllableVoxel::getControlEnergy).sum() / maxYTime;

            }
            centerPositions.add(Point2.build(robot.getCenter()));
            if (listener != null) {
                Snapshot snapshot = new Snapshot(t, (Collection)worldObjects.stream().map(WorldObject::immutable).collect(Collectors.toList()));
                listener.listen(snapshot);
            }
        }
        List<Double> results = new ArrayList(this.metrics.size());
        Iterator var15 = this.metrics.iterator();

        while(var15.hasNext()) {
            it.units.erallab.Jump.Metric metric = (it.units.erallab.Jump.Metric)var15.next();
            double value = 0;
            switch(metric) {
                case CENTER_JUMP:
                    value = (centerPositions.stream().mapToDouble(p -> p.y).max().orElse(0.0D) - centerPositions.get(0).y);
                    break;
                case CONTROL_POWER:
                    value = controlEnergy;
                    break;
                default:
                    throw new IllegalStateException("Unexpected value: " + metric);
            }
            results.add(value);
        }
        return results;
    }

    private static double[][] randomTerrain(int n, double length, double peak, double borderHeight, Random random) {
        double[] xs = new double[n + 2];
        double[] ys = new double[n + 2];
        xs[0] = 0.0D;
        xs[n + 1] = length;
        ys[0] = borderHeight;
        ys[n + 1] = borderHeight;

        for(int i = 1; i < n + 1; ++i) {
            xs[i] = 1.0D + (double)(i - 1) * (length - 2.0D) / (double)n;
            ys[i] = random.nextDouble() * peak;
        }

        return new double[][]{xs, ys};
    }

    public static double[][] createTerrain(String name) {
        Random random = new Random(1L);
        if (name.equals("flat")) {
            return new double[][]{{0.0D, 10.0D, 1990.0D, 2000.0D}, {100.0D, 0.0D, 0.0D, 100.0D}};

        } else if (name.equals("bowl")) {
            return new double[][]{{0.0D, 1.0D, 35.0D, 36.0D, }, {100.0D, 0.0D, 0.0D, 100.0D}};
        } else if (name.startsWith("uneven")) {
            int h = Integer.parseInt(name.replace("uneven", ""));
            return randomTerrain(50, 2000.0D, (double)h, 100.0D, random);
        } else {
            return null;
        }
    }

    public List<it.units.erallab.Jump.Metric> getMetrics() {
        return this.metrics;
    }

    public static enum Metric {
        CENTER_JUMP(true),
        CONTROL_POWER(true);

        private final boolean toMinimize;

        private Metric(boolean toMinimize) {
            this.toMinimize = toMinimize;
        }

        public boolean isToMinimize() {
            return this.toMinimize;
        }
    }
}
