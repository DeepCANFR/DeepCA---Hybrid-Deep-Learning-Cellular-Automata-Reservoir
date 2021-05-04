package it.units.erallab;

import it.units.erallab.hmsrobots.core.objects.ControllableVoxel;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import java.util.EnumSet;

public class Material {
    // voxel made of the soft material
    static final ControllableVoxel softMaterial = new ControllableVoxel(
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
}
