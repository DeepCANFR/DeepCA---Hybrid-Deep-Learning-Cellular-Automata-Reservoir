package it.units.erallab;

import it.units.erallab.hmsrobots.core.controllers.Controller;
import it.units.erallab.hmsrobots.core.controllers.TimeFunctions;
import it.units.erallab.hmsrobots.core.objects.ControllableVoxel;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.util.Grid;
import org.apache.commons.lang3.SerializationUtils;
import org.dyn4j.dynamics.Settings;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public class ThresholdValidation {

    public static void main(String[] args) throws IOException {
        double finalT = 30;
        double pulseDuration = 0.4;
        double minThreshold = 0.00005; // 0.00005 //5e-5
        double maxThreshold = 0.008; // 0.008 //8e-3
        double deltaTau = (maxThreshold - minThreshold)/10;

        Path path = Paths.get("threshold_validation.txt");
        Files.write(path, List.of("grid.side;tau;spatial.size;x;y"), StandardCharsets.UTF_8);

        String serializedSolution = "H4sIAAAAAAAAAOxUv28TMRR+vbYktGppmoIYYCIsSM0NlZBQhjTKXUqka4tyEYJ2KG7iJK58d8Hna68dKvEHsFMEGwMDKwMSLCxFTNCO/AdISF3Y8bv8aAAJFSSkDGfp7OfPz/6+9/x8r77BuC8gw2Q2cJn0s1QQzslmtuX4wtv0FBJIxrNLgtXfn792qXDz8LUGo2UYaalvxwJN+hJmrC2yTXT01C3my1zYBoCE+s6pw9O42DmmIATZRY/w0dHVg0PyfBRGyjDmsz0abbmxM4Z9WzW18dYfVNU8QbPe5hatqUnRc6Xw0IPTu15I+fdnGx+ul29f0UAzIOmQsOSJGrVgsoHjMpUtry5hyWJSjwj0LoHeJ9CRQO8S6L8RZEqnB+VCAfNnlRrtfpcIX5w8WH2rwZQBKSIoqRDJvGUSGpRLoiQ3BKspxF2DOc4cJiMBJIJKnDQNGHOI7xswi0PBbQacCIM4beY21yCFYFHJZX7fP8Is5tK+nwFpxGxWpxZ1m7IVaTBgUlBfMhkglwETfn/dgITfFrizb5UsmO1Ydo00Gh6vK1MVRHqgIEw3cGyKNZH/BFFbvN8ZR/InHaM39o388ULU8m8OsD1dTP6ML8515li6p9XVZcrYVDDC2R7G494RXrg7njj+sl39+FnVgyoCyqlDXVndbVMJqY5UTtymXuQqIbl1SHY9VCSp9YF1JFBxbAvI/9V9Z+xfc9SNF2YBVPVMRzEgRRTD4GI7ED0Npw65l4+/zi8c6U809FCe2v5D2IdpCVN22TA3zHtVs7JSsPpo0VypVgrWRrGyatu99O8LMP/5hQ0+gEG9yHhBQtIo29XCStHEl4zY+H+4fDx3AruZAPsUJgKNNHYXe0Q4uTwcIuJcDFsuYg2xhlhDrGHIf1OxCBiioog19B5H+AMAAP//AwDdV83KQA4AAA==";
        Grid<ControllableVoxel> solution = it.units.erallab.Utils.safelyDeserialize(serializedSolution, Grid.class);

        List<Double> thresholds = new ArrayList<>();
        for (double tau = minThreshold; tau < maxThreshold; tau+=deltaTau) {
            thresholds.add(tau);
        }
        int n = (int) solution.values().stream().filter(Objects::nonNull).count();
        thresholds.add(0.006092750496194226 - 0.00014273597198968677 * n + 9.238294116972325e-07 * n * n);

        for (double tau : thresholds) {
            Grid<ControllableVoxel> body = SerializationUtils.clone(solution);
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
