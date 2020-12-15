package it.units.erallab;

import it.units.erallab.hmsrobots.core.controllers.Controller;
import it.units.erallab.hmsrobots.core.controllers.TimeFunctions;
import it.units.erallab.hmsrobots.core.objects.ControllableVoxel;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.util.Grid;
import org.apache.commons.lang3.SerializationUtils;
import org.dyn4j.dynamics.Settings;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.EnumSet;
import java.util.List;

public class ThresholdTesting {

    public static void main(String[] args) throws IOException {

        double finalT = 30;
        double pulseDuration = 0.4;
        double minThreshold = 0.00005; // 0.00005 //5e-5
        double maxThreshold = 0.008; // 0.008 //8e-3

        int minGridSide = 2;
        int maxGridSide = 10;

        double deltaTau = (maxThreshold - minThreshold)/10;

        Path path = Paths.get("threshold_testing.txt");
        Files.write(path, List.of("grid.side;tau;spatial.size;x;y"), StandardCharsets.UTF_8);

        final ControllableVoxel softMaterial = new ControllableVoxel(
                Voxel.SIDE_LENGTH,
                Voxel.MASS_SIDE_LENGTH_RATIO,
                5d, // low frequency
                Voxel.SPRING_D,
                Voxel.MASS_LINEAR_DAMPING,
                Voxel.MASS_ANGULAR_DAMPING,
                Voxel.FRICTION,
                Voxel.RESTITUTION,
                Voxel.MASS,
                Voxel.LIMIT_CONTRACTION_FLAG,
                Voxel.MASS_COLLISION_FLAG,
                Voxel.AREA_RATIO_MAX_DELTA,
                EnumSet.of(Voxel.SpringScaffolding.SIDE_EXTERNAL, Voxel.SpringScaffolding.CENTRAL_CROSS), // scaffolding partially enabled
                ControllableVoxel.MAX_FORCE,
                ControllableVoxel.ForceMethod.DISTANCE
        );
        for (int n = minGridSide; n < maxGridSide+1; n++) {
            for (double tau = minThreshold; tau < maxThreshold; tau+=deltaTau) {
                Grid<ControllableVoxel> body = Grid.create(n, n, (x,y) -> SerializationUtils.clone(softMaterial));
                // task
                CriticalityEvaluator criticalityEvaluator = new CriticalityEvaluator(
                        finalT, // task duration
                        new Settings(), // default settings for the physics engine
                        tau
                );
                // a pulse controller is applied on each voxel
                for (Grid.Entry<ControllableVoxel> voxel : body) {
                    Controller<ControllableVoxel> pulseController = new TimeFunctions(Grid.create(body.getW(), body.getH(), (x, y) -> (Double t) -> {
                        if (x == voxel.getX() && y == voxel.getY()) {
                            if (t < pulseDuration/2) {
                                return 1.0;
                            } else if (t < pulseDuration) {
                                return -1.0;
                            }
                        }
                        return 0.0;
                    }));
                    int spatialSize = criticalityEvaluator.apply(new Robot<>(pulseController, SerializationUtils.clone(body))).get(0).intValue();
                    Files.write(path, List.of(n+";"+tau+";"+spatialSize+";"+voxel.getX()+";"+voxel.getY()), StandardOpenOption.APPEND);
                }
            }
        }
    }
}
