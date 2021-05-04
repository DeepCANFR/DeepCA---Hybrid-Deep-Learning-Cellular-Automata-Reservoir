package it.units.erallab;

import it.units.erallab.hmsrobots.util.Point2;
import it.units.malelab.jgea.Worker;
import it.units.malelab.jgea.core.Individual;
import it.units.malelab.jgea.core.Problem;
import it.units.malelab.jgea.core.evolver.Evolver;
import it.units.malelab.jgea.core.evolver.StandardEvolver;
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
import java.util.*;
import java.util.concurrent.ExecutionException;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class PowerLawEvolution extends Worker {

    public PowerLawEvolution(String[] args) {
        super(args);
    }

    public static void main(String[] args) {
        new PowerLawEvolution(args);
    }

    @Override
    public void run() {
        int scale = 100;
        int genotypeSize = 20;
        int populationSize = 1000;

        MultiFileListenerFactory<Object, Function<Double, Double>, Double> statsListenerFactory = new MultiFileListenerFactory<>(
                a("dir", "."),
                a("statsFile", null)
        );

        // problem
        Problem<List<Integer>, Double> problem = () -> empiricalFrequencies -> {
            int nonzero = (int) empiricalFrequencies.stream().filter(freq -> freq > 0.0).count();
            if (nonzero < 2) return 0.0;
            // compute the log-log of the distribution
            List<Point2> logLogEmpiricalValues = IntStream.range(0, empiricalFrequencies.size())
                    .mapToObj(i -> Point2.build(i > 0 ? Math.log10(i) : 0, empiricalFrequencies.get(i) > 0.0 ? Math.log10(empiricalFrequencies.get(i)) : 0))
                    .collect(Collectors.toList());
            // linear regression of the log-log distribution
            LinearRegression lr = new LinearRegression(logLogEmpiricalValues);
            double RSquared = 0;
            if (!Double.isNaN(lr.R2())) {
                RSquared = lr.R2(); //(spatialLinearRegression.R2() + temporalLinearRegression.R2())/2;
            }
            // KS statistics
            double ks = BodyOptimization.computeKSStatistics(logLogEmpiricalValues, lr);
            double DSquared = Math.pow(Math.exp(-(0.9 * Math.min(ks, ks) + 0.1 * (ks + ks)/2)), 2d);
            //System.out.println(RSquared+DSquared);
            return RSquared + DSquared + Math.min(1, (empiricalFrequencies.get(0) - empiricalFrequencies.get(empiricalFrequencies.size()-1)));
        };

        // direct mapper
        Function<List<Double>, List<Integer>> mapper = g -> g.stream().map(frequency -> (int) (scale * frequency)).collect(Collectors.toList());

        Evolver<List<Double>, List<Integer>, Double> evolver = new StandardEvolver<>(
                mapper,
                new FixedLengthListFactory<>(genotypeSize, new UniformDoubleFactory(0, 1)),
                PartialComparator.from(Double.class).reversed().comparing(Individual::getFitness), // fitness comparator
                populationSize, // pop size
                Map.of(
                        new UniformCrossover<>(new FixedLengthListFactory<>(genotypeSize, new UniformDoubleFactory(0, 1))), 0.8,
                        new GaussianMutation(0.5), 0.2
                ),
                new Tournament(10), // depends on pop size
                new Worst(), // worst individual dies
                populationSize,
                true
        );

        List<DataCollector<?, ? super List<Double>, ? super Double>> collectors = List.of(
                new Basic(),
                new Population(),
                new Diversity(),
                new BestInfo("%6.4f")
        );

        Listener<? super Object, List<Integer>, ? super Double> listener;
        if (statsListenerFactory.getBaseFileName() == null) {
            listener = listener(collectors.toArray(DataCollector[]::new));
        } else {
            listener = statsListenerFactory.build(collectors.toArray(DataCollector[]::new));
        }

        // optimization
        Collection<List<Integer>> solutions = new ArrayList<>();
        try {
           solutions = evolver.solve(
                    Misc.cached(problem.getFitnessFunction(), 10000),
                    new Iterations(200),
                    new Random(1),
                    executorService,
                    listener);
        } catch (InterruptedException e) {
            e.printStackTrace();
        } catch (ExecutionException e) {
            e.printStackTrace();
        }
        // print one solution
        if (solutions.size() > 0) {
            List<Integer> best = (List<Integer>) solutions.toArray()[0];
            for (int i = 0; i < best.size(); i++) {
                System.out.print(best.get(i)+", ");
            }
            System.out.println();
        }
    }
}