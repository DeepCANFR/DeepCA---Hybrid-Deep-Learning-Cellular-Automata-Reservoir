package it.units.erallab;

import it.units.erallab.hmsrobots.core.objects.ControllableVoxel;
import it.units.erallab.hmsrobots.util.Grid;
import org.apache.commons.lang3.SerializationUtils;
import org.apache.commons.math3.util.Pair;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static it.units.erallab.RobotValidator.validateBodyCriticality;
import static it.units.erallab.Utils.*;

public class CriticalityMeasure {

    public static void main(String[] args) throws FileNotFoundException {
        int robotVoxels = 20;
        int gridW = 20;
        int gridH = 20;
        int seed = 0;
        Random random = new Random(seed);
        // generate all bodies
        List<Pair<String, Grid<ControllableVoxel>>> bodies = new ArrayList<>();
        // add optimized bodies
        int index = 0;
        for (String serializedOptimizedBody : Utils.optimizedBodies) {
            bodies.add(new Pair("opt-"+index, Utils.safelyDeserialize(serializedOptimizedBody, Grid.class)));
            index ++;
        }
        // 10 pseudo-random bodies
        for (int j = 0; j < 10; j++) {
            bodies.add(new Pair("pseudrnd-"+j, generatePseudoRandomBody(robotVoxels, Math.max(gridW, gridH), random, SerializationUtils.clone(Material.softMaterial))));
        }
        // 10 random bodies
        for (int j = 0; j < 10; j++) {
            bodies.add(new Pair("rnd-"+j, generateRandomBody(robotVoxels, Math.max(gridW, gridH), random, SerializationUtils.clone(Material.softMaterial))));
        }
        // box
        bodies.add(new Pair("box", Grid.create(5, 4, (x, y) -> SerializationUtils.clone(Material.softMaterial))));
        // worm
        bodies.add(new Pair("worm", Grid.create(10, 2, (x, y) -> SerializationUtils.clone(Material.softMaterial))));
        // biped
        bodies.add(new Pair("biped", Grid.create(6,4, (x, y) -> {
            if ((y > 1) || (x < 2 || x > 3)) {
                return SerializationUtils.clone(Material.softMaterial);
            } else {
                return null;
            }
        })));
        // reversed T
        bodies.add(new Pair("revT", Grid.create(6,6, (x, y) -> {
            if ((y < 2) || (x > 1 && x < 4)) {
                return SerializationUtils.clone(Material.softMaterial);
            } else {
                return null;
            }
        })));


        File csvCriticality = new File("bodiesCriticality.csv");
        String sep = "\t";
        PrintWriter writer = new PrintWriter(new FileOutputStream(csvCriticality, false));
        writer.write("name"+sep+"criticality"+sep+"body"+"\n");
        writer.close();

        for (Pair<String, Grid<ControllableVoxel>> nameBodyPair : bodies) {
            String name = nameBodyPair.getFirst();
            Grid<ControllableVoxel> body = nameBodyPair.getSecond();
            // test criticality for each body
            double score = validateBodyCriticality(body);
            System.out.println(name+" => "+score);
            System.out.println(safelySerialize(body));
            // draw robot
            String bodyRepresentationString = bodyToString(body);
            try (PrintWriter awriter = new PrintWriter(new FileOutputStream(csvCriticality ,true))) {
                awriter.write(name+sep+score+sep+bodyRepresentationString+"\n");
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }
        }
    }
}
