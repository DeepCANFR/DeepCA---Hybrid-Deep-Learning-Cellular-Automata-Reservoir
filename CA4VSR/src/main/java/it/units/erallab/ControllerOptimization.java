package it.units.erallab;

import com.google.common.collect.Lists;
import it.units.erallab.hmsrobots.core.controllers.CentralizedSensing;
import it.units.erallab.hmsrobots.core.controllers.Controller;
import it.units.erallab.hmsrobots.core.controllers.MultiLayerPerceptron;
import it.units.erallab.hmsrobots.core.controllers.TimeFunctions;
import it.units.erallab.hmsrobots.core.objects.ControllableVoxel;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.SensingVoxel;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.core.sensors.*;
import it.units.erallab.hmsrobots.tasks.Locomotion;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.malelab.jgea.Worker;
import it.units.malelab.jgea.core.Individual;
import it.units.malelab.jgea.core.Problem;
import it.units.malelab.jgea.core.evolver.CMAESEvolver;
import it.units.malelab.jgea.core.evolver.Evolver;
import it.units.malelab.jgea.core.evolver.StandardEvolver;
import it.units.malelab.jgea.core.evolver.stopcondition.Births;
import it.units.malelab.jgea.core.evolver.stopcondition.Iterations;
import it.units.malelab.jgea.core.listener.Listener;
import it.units.malelab.jgea.core.listener.MultiFileListenerFactory;
import it.units.malelab.jgea.core.listener.collector.*;
import it.units.malelab.jgea.core.order.PartialComparator;
import it.units.malelab.jgea.core.selector.Tournament;
import it.units.malelab.jgea.core.selector.Worst;
import it.units.malelab.jgea.core.util.Misc;
import it.units.malelab.jgea.representation.sequence.FixedLengthListFactory;
import it.units.malelab.jgea.representation.sequence.UniformCrossover;
import it.units.malelab.jgea.representation.sequence.numeric.GaussianMutation;
import it.units.malelab.jgea.representation.sequence.numeric.UniformDoubleFactory;
import org.apache.commons.lang3.SerializationUtils;
import org.dyn4j.dynamics.Settings;
import java.util.*;
import java.util.concurrent.ExecutionException;
import java.util.function.Function;
import java.util.stream.IntStream;

import static it.units.malelab.jgea.core.util.Args.i;

public class ControllerOptimization extends Worker {

    public ControllerOptimization(String[] args) {
        super(args);
    }


    public static void main(String[] args) {
        new ControllerOptimization(args);
    }

    public void run() {
        String bodyType = a("bodyType", "serialized");
        String controller = a("controller", "centralized");
        int robotIndex = i(a("robotIndex", "0"));
        int gridW = i(a("gridW", "20"));
        int gridH = i(a("gridH", "20"));
        int robotVoxels = i(a("robotVoxels", "20"));
        int randomSeed = i(a("randomSeed", "666"));
        Random random = new Random(randomSeed);
        int cacheSize = 10000;
        double time = 20;
        int births = i(a("births", "10000"));
        String taskType = a("taskType", "jump");

        Grid<ControllableVoxel> body = null;

        List<String> bodies = null;
        if (bodyType.equals("pseudorandom")) {
            bodies = Utils.pseudoRandomBodies;
        } else if (bodyType.equals("random")) {
            bodies = Utils.randomBodies;
        } else if (bodyType.equals("serliazied")) {
            bodies = Utils.optimizedBodies;
        }
        if (bodies != null) {
            body = it.units.erallab.Utils.safelyDeserialize(bodies.get(robotIndex), Grid.class);
            for (int x = 0; x < body.getW(); x++) {
                for (int y = 0; y < body.getH(); y++) {
                    if (body.get(x, y) != null) {
                        body.set(x, y, new ControllableVoxel());
                    }
                }
            }
        }

        if (bodyType.equals("box")) {
            body = Grid.create(gridW, gridH, (x, y) -> SerializationUtils.clone(new ControllableVoxel()));
        } else if (bodyType.equals("worm")) {
            body = Grid.create(10, 2, (x, y) -> SerializationUtils.clone(new ControllableVoxel()));
        }  else if (bodyType.equals("biped")) {
            body = Grid.create(6,4, (x, y) -> {
                if ((y > 1) || (x < 2 || x > 3)) {
                    return SerializationUtils.clone(new ControllableVoxel());
                } else {
                    return null;
                }
            });
        }  else if (bodyType.equals("reversedT")) {
            body = Grid.create(6,6, (x, y) -> {
                if ((y < 2) || (x > 1 && x < 4)) {
                    return SerializationUtils.clone(new ControllableVoxel());
                } else {
                    return null;
                }
            });
        }

        int genotypeSize;
        if (controller.equals("phase")) {
            genotypeSize = body.getW() * body.getH();
        } else {
            Grid<ControllableVoxel> finalBody = body;
            Grid<SensingVoxel> mlpBody = Grid.create(body.getW(), body.getH(), (x, y) -> {
                SensingVoxel sensingVoxel = null;
                if (finalBody.get(x, y) != null) {
                    sensingVoxel = new SensingVoxel(List.of(
                            new Touch(),
                            new Normalization(new Velocity(true, 5d, Velocity.Axis.X, Velocity.Axis.Y)),
                            new Normalization(new AreaRatio())
                    ));
                }
                return sensingVoxel;
            });
            CentralizedSensing cBrain  = new CentralizedSensing<>(SerializationUtils.clone(mlpBody));
            genotypeSize = new MultiLayerPerceptron(MultiLayerPerceptron.ActivationFunction.TANH, cBrain.nOfInputs(), new int[]{(int) (cBrain.nOfInputs() * 0.65d)}, cBrain.nOfOutputs()).getParams().length;
        }

        MultiFileListenerFactory<Object, Robot<? extends Voxel>, Double> statsListenerFactory = new MultiFileListenerFactory<>(
                a("dir", "."),
                a("statsFile", null)
        );

        Function<?, ?> task = null;

        if (taskType.equals("locomotion")) {
            task = Misc.cached(new Locomotion(
                    time,
                    Locomotion.createTerrain("flat"),
                    Lists.newArrayList(Locomotion.Metric.TRAVEL_X_VELOCITY, Locomotion.Metric.CONTROL_POWER),
                    new Settings()
            ), cacheSize);
        } else if (taskType.equals("hiking")) {
            task = Misc.cached(new Locomotion(
                    time,
                    Utils.createHillyTerrain(1.0,1.0,0),
                    Lists.newArrayList(Locomotion.Metric.TRAVEL_X_VELOCITY, Locomotion.Metric.CONTROL_POWER),
                    new Settings()
            ), cacheSize);
        } else if (taskType.equals("escape")) {
            task = Misc.cached(new Escape(
                    40.0,
                    Lists.newArrayList(Locomotion.Metric.TRAVEL_X_VELOCITY, Locomotion.Metric.CONTROL_POWER),
                    new Settings()
            ), cacheSize);
        } else if (taskType.equals("jump")) {
            task = Misc.cached(new Jump(
                    time,
                    Jump.createTerrain("bowl"),
                    1.0,
                    Lists.newArrayList(Jump.Metric.CENTER_JUMP, Jump.Metric.CONTROL_POWER),
                    new Settings()
            ), cacheSize);
        }

        Function<Robot<? extends Voxel>, ?> finalTask = (Function<Robot<? extends Voxel>, ?>) task;
        Problem<Robot<? extends Voxel>, Double> problem = () -> robot -> {
            List<Double> results = (List<Double>) finalTask.apply(robot);
            //return results.get(0)*(1.0/(1.0 + results.get(1)*time*time));
            return results.get(0);
        };

        Grid<ControllableVoxel> finalBody1 = body;
        Function<List<Double>, Robot<? extends Voxel>> mapper = g -> {
            Controller<? extends Voxel> brain = null;
            Robot<? extends Voxel> robot = null;
            if (controller.equals("phase")) {
                // each element of g becomes a phase
                brain = new TimeFunctions(
                        Grid.create(finalBody1.getW(), finalBody1.getH(), (x, y) -> (Double t) -> Math.sin(-2 * Math.PI * t + Math.PI * g.get(x + y * finalBody1.getW())))
                );
                robot = new Robot<>((Controller<? super ControllableVoxel>) brain, SerializationUtils.clone(finalBody1));
            } else if (controller.equals("centralized")) {
                // convert body to sensing body
                Grid<SensingVoxel> sensingBody = Grid.create(finalBody1.getW(), finalBody1.getH(), (x, y) -> {
                    SensingVoxel sensingVoxel = null;
                    if (finalBody1.get(x, y) != null) {
                        sensingVoxel = new SensingVoxel(List.of(
                                new Touch(),
                                new Normalization(new Velocity(true, 5d, Velocity.Axis.X, Velocity.Axis.Y)),
                                new Normalization(new AreaRatio())
                                ));
                    }
                    return sensingVoxel;
                });
                CentralizedSensing centralizedBrain  = new CentralizedSensing<>(SerializationUtils.clone(sensingBody));
                MultiLayerPerceptron mlp = new MultiLayerPerceptron(
                        MultiLayerPerceptron.ActivationFunction.TANH,
                        centralizedBrain.nOfInputs(),
                        // 0.65d
                        new int[]{(int) (centralizedBrain.nOfInputs() * 0.65d)}, // hidden layers
                        centralizedBrain.nOfOutputs()
                );
                double[] ws = mlp.getParams();
                IntStream.range(0, ws.length).forEach(i -> ws[i] = g.get(i));
                mlp.setParams(ws);
                centralizedBrain.setFunction(mlp);
                robot = new Robot<>(centralizedBrain, SerializationUtils.clone(sensingBody));
            }
            return robot;
        };

        // CMA-ES evolver: https://en.wikipedia.org/wiki/CMA-ES
        Evolver<List<Double>, Robot<? extends Voxel>, Double> evolver = new CMAESEvolver<>(
                mapper,
                new FixedLengthListFactory<>(genotypeSize, new UniformDoubleFactory(0, 1)),
                PartialComparator.from(Double.class).reversed().comparing(Individual::getFitness),
                0,
                1
        );

        // standard evolver
        Evolver<List<Double>, Robot<? extends Voxel>, Double> directEvolver = new StandardEvolver<>(
                mapper,
                new FixedLengthListFactory<>(genotypeSize, new UniformDoubleFactory(0, 1)),
                PartialComparator.from(Double.class).reversed().comparing(Individual::getFitness), // fitness comparator
                1000, // pop size
                Map.of(
                        new GaussianMutation(0.01), 0.2d,
                        new UniformCrossover<>(new FixedLengthListFactory<>(genotypeSize, new UniformDoubleFactory(0, 1))), 0.8d
                ),
                new Tournament(10), // depends on pop size
                new Worst(), // worst individual dies
                1000,
                true
        );

        List<DataCollector<?, ? super Robot<? extends Voxel>, ? super Double>> collectors = List.of(
                new Basic(),
                new Population(),
                new Diversity(),
                new BestInfo("%8.6f"),
                new FunctionOfOneBest<>(i -> List.of(
                        new Item("serialized.robot", it.units.erallab.Utils.safelySerialize(i.getSolution()), "%s")
                ))
        );
        Listener<? super Object, ? super Robot<? extends Voxel>, ? super Double> listener;
        if (statsListenerFactory.getBaseFileName() == null) {
            listener = listener(collectors.toArray(DataCollector[]::new));
        } else {
            listener = statsListenerFactory.build(collectors.toArray(DataCollector[]::new));
        }
        try {
            if (controller.equals("phase")) {
                evolver.solve(
                        Misc.cached(problem.getFitnessFunction(), cacheSize),
                        new Births(births),
                        new Random(randomSeed),
                        executorService,
                        listener
                );
            } else {
                directEvolver.solve(
                        Misc.cached(problem.getFitnessFunction(), cacheSize),
                        new Iterations(100),
                        new Random(randomSeed),
                        executorService,
                        listener
                );
            }
        } catch (InterruptedException e) {
            e.printStackTrace();
        } catch (ExecutionException e) {
            e.printStackTrace();
        }
    }
}
