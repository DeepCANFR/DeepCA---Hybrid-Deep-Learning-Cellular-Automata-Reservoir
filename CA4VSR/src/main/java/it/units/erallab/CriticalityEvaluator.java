package it.units.erallab;

import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.WorldObject;
import it.units.erallab.hmsrobots.core.objects.immutable.Snapshot;
import it.units.erallab.hmsrobots.tasks.AbstractTask;
import it.units.erallab.hmsrobots.util.BoundingBox;
import it.units.erallab.hmsrobots.viewers.SnapshotListener;
import org.dyn4j.dynamics.Settings;
import org.dyn4j.dynamics.World;
import org.dyn4j.geometry.Vector2;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class CriticalityEvaluator extends AbstractTask<Robot<?>, List<Double>> {
    private final double finalT;
    private final double initialPlacement;
    private double threshold;
    private boolean dynamicThreshold;

    public CriticalityEvaluator(double finalT, Settings settings) {
        super(settings);
        this.finalT = finalT;
        this.initialPlacement = 1.0D;
        this.dynamicThreshold = true;
    }

    public CriticalityEvaluator(double finalT, Settings settings, double threshold) {
        super(settings);
        this.finalT = finalT;
        this.initialPlacement = 1.0D;
        this.dynamicThreshold = false;
        this.threshold = threshold;
    }

    public List<Double> apply(Robot<?> robot, SnapshotListener listener) {

        int voxelGridSize = robot.getVoxels().getW()*robot.getVoxels().getH();

        int voxels = (int) robot.getVoxels().values().stream().filter(Objects::nonNull).count();

        if (this.dynamicThreshold) {
            this.threshold = 0.006092750496194226 - 0.00014273597198968677 * voxels +  9.238294116972325e-07 * voxels * voxels;
        }

        Object[] voxelsPreviousArea = null;
        Object[] voxelsCurrentArea;
        int[] avalanchedVoxels = new int[voxelGridSize];
        int avalanchesTemporalExtension = 0;

        World world = new World();
        // disable gravity
        world.setGravity(new Vector2(0d, 0d));

        world.setSettings(this.settings);
        List<WorldObject> worldObjects = new ArrayList();

        BoundingBox boundingBox = robot.boundingBox();
        robot.translate(new Vector2(this.initialPlacement - boundingBox.min.x, 0.0D));
        robot.addTo(world);
        worldObjects.add(robot);

        double t = 0.0D;
        while (t < this.finalT) {
            t += this.settings.getStepFrequency();
            world.step(1);
            robot.act(t);

            // self-organized criticality
            voxelsCurrentArea = robot.getVoxels().values().stream()
                    .filter(Objects::nonNull)
                    .map(it.units.erallab.hmsrobots.core.objects.Voxel::getAreaRatio)
                    .collect(Collectors.toList()).toArray();

            if (voxelsPreviousArea != null) {
                Object[] finalPreviousAreas = voxelsPreviousArea;
                Object[] finalCurrentAreas = voxelsCurrentArea;
                int[] activeVoxelsIndex = IntStream.range(0, voxelsPreviousArea.length)
                        .filter(i -> Math.abs((double) finalPreviousAreas[i] - (double) finalCurrentAreas[i]) > threshold)
                        .toArray();

                if (activeVoxelsIndex.length == 0) {
                    break;
                } else {
                    avalanchesTemporalExtension += 1;
                    // avalanche spatial extension
                    Arrays.stream(activeVoxelsIndex)
                            .filter(i -> avalanchedVoxels[i] == 0)
                            .forEach(i -> avalanchedVoxels[i] = 1);
                }
            }
            voxelsPreviousArea = voxelsCurrentArea;

            // this saves the robot info during the simulation
            if (listener != null) {
                Snapshot snapshot = new Snapshot(t, (Collection) worldObjects.stream().map(WorldObject::immutable).collect(Collectors.toList()));
                listener.listen(snapshot);
            }
        }
        return List.of((double)Arrays.stream(avalanchedVoxels).sum(), (double)(avalanchesTemporalExtension));
    }
}
