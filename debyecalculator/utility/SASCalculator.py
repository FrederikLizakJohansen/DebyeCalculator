#!/usr/bin/env python

"""\
Support for SAS calculation from Debye Scattering Equation.
"""

import numpy

from diffpy.srreal.pdfcalculator import DebyePDFCalculator
from diffpy.srreal.sfaverage import SFAverage
from diffpy.srfit.pdf.basepdfgenerator import BasePDFGenerator

# ----------------------------------------------------------------------------

class SASCalculator(DebyePDFCalculator):

    _iqtot = None
    sfavg = None


    def __init__(self, **kwargs):
        '''Create a new instance of SASCalculator.
        Keyword arguments can be used to configure the calculator properties,
        for example:

        sasc = SASCalculator(qmax=20, rmax=40)

        Raise ValueError for invalid keyword argument.
        '''
        # set SAS defaults for the qgrid here
        dbkw = dict(qmin=0.0, qmax=5.0, qstep=0.02, rmax=50, rstep=2)
        dbkw.update(kwargs)
        DebyePDFCalculator.__init__(self, **dbkw)
        return


    @property
    def iqtot(self):
        'Un-normalized intensities for SAS at the ggrid points.'
        return self._iqtot

    @property
    def iq(self):
        'Normalized intensities for SAS at the ggrid points.'
        return self._iqtot / self.sfavg.count

    @property
    def sq(self):
        'Unscaled S(Q) at the ggrid points.'
        rv = (self.iq - self.sfavg.f2avg) / (self.sfavg.f1avg ** 2) + 1
        return rv

    @property
    def fq(self):
        'Unscaled F(Q) at the ggrid points.'
        rv = (self.sq - 1) * self.qgrid
        return rv


    def __call__(self, structure=None, **kwargs):
        '''Calculate SAS for the given structure as an (r, Itot) tuple.
        Keyword arguments can be used to configure calculator attributes,
        these override any properties that may be passed from the structure,
        such as spdiameter.

        structure    -- a structure object to be evaluated.  Reuse the last
                        structure when None.
        kwargs       -- optional parameter settings for this calculator

        Example:    sascalc(structure, qmax=2, rmax=50)

        Return a tuple of (qgrid, iqtot) numpy arrays.
        '''
        from diffpy.srreal.wraputils import setattrFromKeywordArguments
        setattrFromKeywordArguments(self, **kwargs)
        self.eval(structure)
        setattrFromKeywordArguments(self, **kwargs)
        rv = (self.qgrid, self.iqtot)
        return rv


    def eval(self, structure=None):
        '''Evaluate Debye Scattering Equation for the given or last structure.
        '''
        DebyePDFCalculator.eval(self, structure)
        qa = self.qgrid
        adpt = self.getStructure()
        tbl = self.scatteringfactortable
        sfavg = SFAverage.fromStructure(adpt, tbl, qa)
        idxhi = (qa > 1e-8)
        idxlo = (~idxhi).nonzero()
        iqtot = numpy.zeros_like(qa)
        iqtot[idxlo] = sfavg.f1sum[idxlo] ** 2
        iqtot[idxhi] = self.value[idxhi] / qa[idxhi] + sfavg.f2sum[idxhi]
        self._iqtot = iqtot
        self.sfavg = sfavg
        return

# End of class SASCalculator

# ----------------------------------------------------------------------------

class DBSASGenerator(BasePDFGenerator):

    _useadp = True
    _adpparnames = set('''
        B11 B12 B13 B21 B22 B23 B31 B32 B33 Biso
        U11 U12 U13 U21 U22 U23 U31 U32 U33 Uiso
        '''.split())


    def __init__(self, name="dbsas"):
        BasePDFGenerator.__init__(self, name)
        sc = SASCalculator()
        self._setCalculator(sc)
        self.removeParameter(self.delta1)
        self.removeParameter(self.delta2)
        self.removeParameter(self.qbroad)
        self.removeParameter(self.qdamp)
        return


    @property
    def rmax(self):
        "float : pair-distance upper bound for Debye summation."
        return self._calc.rmax

    @rmax.setter
    def rmax(self, rmax):
        self._calc.rmax = rmax
        return


    @property
    def useadp(self):
        """bool : flag for recalculating the value when ADP parameters change.

        Use ``False`` in SAS regime, when ADP parameters have little effect on
        the generated profile.  The default is ``True``.
        """
        return self._useadp

    @useadp.setter
    def useadp(self, flag):
        if bool(flag) is not self._useadp:
            self._useadp = bool(flag)
            self._flush(other=(self,))
        return


    def _prepare(self, q):
        """Prepare the calculator when a new q-grid is passed."""
        # store it in _lastr which is consulted in BasePDFGenerator.__call__
        self._lastr = q
        self._calc.qstep = q[1] - q[0]
        self._calc.qmin = q[0]
        self._calc.qmax = q[-1] + 0.5*self._calc.qstep
        return


    def _flush(self, other):
        '''Ignore changes in ADP values when `useadp` is ``False``.
        '''
        ignore = (not self.useadp and isinstance(other, tuple) and
                  len(other) > 1 and other[-1].name in self._adpparnames)
        if not ignore:
            BasePDFGenerator._flush(self, other)
        return


    def __call__(self, q):
        """Calculate the I(Q) intensity from SAS."""
        # SASCalculator ignores the scale, so we add it in here
        yout = BasePDFGenerator.__call__(self, q)
        yout *= self.scale.value
        return yout

# End class DBSASGenerator
