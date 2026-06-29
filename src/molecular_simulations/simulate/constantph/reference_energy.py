import numpy as np
from openmm.unit import MOLAR_GAS_CONSTANT_R, is_quantity, kelvin, kilojoules_per_mole
from scipy.optimize import curve_fit


class ReferenceEnergyFinder:
    def __init__(self, model, pKa, temperature):
        """Construct a ReferenceEnergyFinder.

        Args:
            model: ConstantPH model for which to determine reference energies.
                It must contain a single titratable residue with exactly two
                states. It does not matter what pH or reference energies were
                specified when it was created, because they will both be
                overwritten.
            pKa: The experimental pKa of the titratable residue. Reference
                energies will be chosen to match it.
            temperature: The temperature (openmm.unit.Quantity) at which the
                simulation will be run.
        """
        if len(model.titrations) != 1:
            raise ValueError(
                'The model compound must contain a single titratable residue'
            )
        self.model = model
        self.pKa = pKa
        if not is_quantity(temperature):
            temperature = temperature * kelvin
        self.temperature = temperature
        self.residueIndex = next(iter(model.titrations.keys()))
        self.titration = model.titrations[self.residueIndex]
        if len(self.titration.explicitStates) != 2:
            raise ValueError('Only residues with two states are currently supported')

    def findReferenceEnergies(self, iterations=20000, substeps=20):
        """Compute the reference energies for the states of the model compound.

        On exit, they will be stored in the ConstantPH object.

        Args:
            iterations: The number of Monte Carlo moves to attempt. The larger
                the number, the more tightly converged the results will be.
            substeps: The number of dynamics steps to integrate between Monte
                Carlo moves.
        """
        # Find an initial estimate of the reference energies just by computing the potential
        # energies of the states.

        self.model.setResidueState(self.residueIndex, 0)
        energy0 = self.model.implicitContext.getState(
            getEnergy=True
        ).getPotentialEnergy()
        self.model.setResidueState(self.residueIndex, 1)
        energy1 = self.model.implicitContext.getState(
            getEnergy=True
        ).getPotentialEnergy()
        deltaN = (
            self.titration.implicitStates[1].numHydrogens
            - self.titration.implicitStates[0].numHydrogens
        )
        scale = MOLAR_GAS_CONSTANT_R * self.temperature * deltaN * np.log(10.0)
        self.titration.referenceEnergies = [
            0.0 * kilojoules_per_mole,
            energy1 - energy0,
        ]
        self.model.simulation.minimizeEnergy()
        self.model.simulation.context.setVelocitiesToTemperature(self.temperature)

        # If our initial estimate is exact, the fractions should be equal at pH 0.  Since it probably
        # isn't, simulate it at various pHs to refine the estimate.

        while True:
            self.model.setPH([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
            for _ in range(1000):
                self.model.simulation.step(substeps)
                self.model.attemptMCStep(self.temperature)
            fractions = [[] for _ in range(len(self.model.pH))]
            for _ in range(iterations):
                self.model.simulation.step(substeps)
                self.model.attemptMCStep(self.temperature)
                fractions[self.model.currentPHIndex].append(
                    1.0
                    if self.titration.protonatedIndex == self.titration.currentIndex
                    else 0.0
                )

            # Fit a curve to the data to better estimate when the fraction is exactly 0.5,
            # and compute the reference energy based on it.

            x = []
            y = []
            for i in range(len(fractions)):
                if len(fractions[i]) > 0:
                    x.append(self.model.pH[i])
                    y.append(np.average(fractions[i]))

            def f(ph, pka):
                return 1 / (1 + 10 ** (ph - pka))

            popt, _pcov = curve_fit(f, x, y, [0.0])
            root = popt[0]
            if root > -2 and root < 2:
                self.titration.referenceEnergies[1] += scale * (self.pKa - root)
                break
            self.titration.referenceEnergies[1] -= scale * root
