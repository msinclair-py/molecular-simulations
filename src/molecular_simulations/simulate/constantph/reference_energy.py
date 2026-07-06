import numpy as np
from openmm.unit import MOLAR_GAS_CONSTANT_R, is_quantity, kelvin, kilojoules_per_mole
from scipy.optimize import curve_fit


def hill_equation(pH, pKa, n):
    """Fraction protonated at a given pH (Hill form; n=1 is Henderson-Hasselbalch)."""
    return 1.0 / (1.0 + 10.0 ** (n * (np.asarray(pH) - pKa)))


def fit_titration_midpoint(pH, fraction, pKa0=7.0):
    """Fit a titration curve and return its midpoint pKa and Hill coefficient.

    Uses a bounded Hill fit, which is far more robust than an unbounded one (the
    latter can wander and exhaust ``maxfev``). If the fit still fails, falls back
    to linear interpolation of the pH where the fraction crosses 0.5.

    Args:
        pH: Sequence of pH values.
        fraction: Fraction protonated at each pH.
        pKa0: Initial guess for the midpoint.

    Returns:
        Tuple ``(midpoint_pKa, hill_n)``. ``hill_n`` is NaN when the fallback
        interpolation was used.
    """
    x = np.asarray(pH, dtype=float)
    y = np.asarray(fraction, dtype=float)
    try:
        popt, _ = curve_fit(
            hill_equation,
            x,
            y,
            p0=[pKa0, 1.0],
            bounds=([x.min() - 5.0, 0.1], [x.max() + 5.0, 5.0]),
            maxfev=20000,
        )
        return float(popt[0]), float(popt[1])
    except (RuntimeError, ValueError):
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        if ys[0] >= 0.5 >= ys[-1]:  # curve brackets 0.5: interpolate the crossing
            return float(np.interp(0.5, ys[::-1], xs[::-1])), float('nan')
        # midpoint lies outside the sampled ladder; report the nearer edge
        return float(xs[0] if ys.mean() < 0.5 else xs[-1]), float('nan')


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

    def _sampleMidpoint(self, ladder, iterations, substeps):
        """Titrate on an absolute pH ladder and return (midpoint pKa, Hill n).

        Runs simulated tempering across ``ladder``, records the fraction of
        samples in the protonated state at each pH, and fits the Hill equation.

        Args:
            ladder: List of pH values to sample.
            iterations: Number of Monte Carlo moves to collect statistics over.
            substeps: Dynamics steps between Monte Carlo moves.

        Returns:
            Tuple ``(pKa, n)`` from the Hill fit to the titration curve.
        """
        self.model.setPH(list(ladder))
        for _ in range(1000):  # let the Wang-Landau weights converge
            self.model.simulation.step(substeps)
            self.model.attemptMCStep(self.temperature)
        fractions = [[] for _ in ladder]
        for _ in range(iterations):
            self.model.simulation.step(substeps)
            self.model.attemptMCStep(self.temperature)
            fractions[self.model.currentPHIndex].append(
                1.0
                if self.titration.currentIndex == self.titration.protonatedIndex
                else 0.0
            )
        x = [ladder[i] for i in range(len(ladder)) if fractions[i]]
        y = [np.mean(fractions[i]) for i in range(len(ladder)) if fractions[i]]
        return fit_titration_midpoint(x, y, pKa0=self.pKa)

    def findReferenceEnergiesIterative(
        self,
        iterations=8000,
        substeps=20,
        halfwidth=3.0,
        tol=0.1,
        max_rounds=5,
    ):
        """Calibrate reference energies so an independent titration hits the pKa.

        :meth:`findReferenceEnergies` applies a single linear correction, which
        can leave a residual error of several tenths of a pKa unit because its
        estimate and a fresh titration sample different pH ladders. This method
        adds Newton refinement on top: it runs a validation titration on an
        absolute ladder centered at the target pKa and shifts the reference
        energy by ``scale * (pKa - midpoint)`` until the midpoint converges. The
        response is linear (``d(midpoint)/d(ref) = 1/scale``), so it converges in
        a couple of rounds.

        Args:
            iterations: Monte Carlo moves per titration.
            substeps: Dynamics steps between Monte Carlo moves.
            halfwidth: The ladder spans ``pKa +/- halfwidth`` in 1-unit steps.
            tol: Convergence tolerance on ``|midpoint - pKa|`` in pKa units.
            max_rounds: Maximum refinement rounds.

        Returns:
            List of ``(reference_energy_kJ, midpoint, hill_n)`` per round, in
            order. The final reference energy is stored on the ConstantPH object.
        """
        deltaN = (
            self.titration.implicitStates[1].numHydrogens
            - self.titration.implicitStates[0].numHydrogens
        )
        scale = MOLAR_GAS_CONSTANT_R * self.temperature * deltaN * np.log(10.0)

        # Initial estimate from the single-conformation implicit energies. Setting
        # ref = (E_prot - E_deprot) + scale * pKa places the titration midpoint at
        # the target pKa to the extent the single point approximates the ensemble
        # average; the Newton refinement below corrects the remainder. This avoids
        # the fixed [-3, 3] ladder in findReferenceEnergies, which fails to fit
        # when every pH bin is fully (de)protonated.
        self.model.setResidueState(self.residueIndex, 0)
        e0 = self.model.implicitContext.getState(getEnergy=True).getPotentialEnergy()
        self.model.setResidueState(self.residueIndex, 1)
        e1 = self.model.implicitContext.getState(getEnergy=True).getPotentialEnergy()
        self.titration.referenceEnergies = [
            0.0 * kilojoules_per_mole,
            (e1 - e0) + scale * self.pKa,
        ]
        self.model.simulation.minimizeEnergy()
        self.model.simulation.context.setVelocitiesToTemperature(self.temperature)

        ladder = [self.pKa - halfwidth + i for i in range(int(2 * halfwidth) + 1)]
        history = []
        for _ in range(max_rounds):
            midpoint, n = self._sampleMidpoint(ladder, iterations, substeps)
            ref_kJ = self.titration.referenceEnergies[1].value_in_unit(
                kilojoules_per_mole
            )
            history.append((float(ref_kJ), midpoint, n))
            if abs(midpoint - self.pKa) < tol:
                break
            self.titration.referenceEnergies[1] += scale * (self.pKa - midpoint)
        return history
