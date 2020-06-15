#!/usr/bin/env python
##############################################################################
#
# pyobjcryst
#
# File coded by:    Vincent Favre-Nicolin
#
# See AUTHORS.txt for a list of people who contributed.
# See LICENSE.txt for license information.
#
##############################################################################

"""Python wrapping of PowderPattern.h

See the online ObjCryst++ documentation (http://vincefn.net/ObjCryst/).

Changes from ObjCryst::PowderPattern::
        In development !
"""

import urllib
import numpy as np

__all__ = ["PowderPattern", "CreatePowderPatternFromCIF",
           "PowderPatternBackground", "PowderPatternComponent",
           "PowderPatternDiffraction", "ReflectionProfileType",
           "gPowderPatternRegistry", "SpaceGroupExplorer"]

from pyobjcryst._pyobjcryst import PowderPattern as PowderPattern_objcryst
from pyobjcryst._pyobjcryst import CreatePowderPatternFromCIF as CreatePowderPatternFromCIF_orig
from pyobjcryst._pyobjcryst import PowderPatternBackground
from pyobjcryst._pyobjcryst import PowderPatternComponent
from pyobjcryst._pyobjcryst import PowderPatternDiffraction
from pyobjcryst._pyobjcryst import ReflectionProfileType
from pyobjcryst._pyobjcryst import gPowderPatternRegistry
from pyobjcryst._pyobjcryst import LSQ
from pyobjcryst.refinableobj import refpartype_scattdata_background
from pyobjcryst._pyobjcryst import SpaceGroupExplorer


class PowderPattern(PowderPattern_objcryst):

    def __init__(self):
        super(PowderPattern, self).__init__()
        self._plot_fig = None
        self._plot_xlim = None
        self._plot_ylim = None
        self._plot_diff = False
        self._plot_hkl = False
        # xlim last time we called
        self._plot_xlim_plot = None

    def UpdateDisplay(self):
        if self._plot_fig is not None:
            import matplotlib.pyplot as plt
            if 'inline' not in plt.get_backend():
                if plt.fignum_exists(self._plot_fig.number):
                    self.plot()

    def plot(self, diff=None, hkl=None, figsize=(9, 4), fontsize_hkl=6, reset=False, **kwargs):
        """
        Show the powder pattern in a plot using matplotlib
        :param diff: if True, also show the difference plot
        :param hkl: if True, print the hkl values
        :param figsize: the figure size passed to matplotlib
        :param fontsize_hkl: fontsize for hkl coordinates
        :param reset: if True, will reset the x and y limits, and remove the flags to plot
                      the difference and hkl unless the options are set at the same time.
        :param kwargs: additional keyword arguments:
                       fig=None will force creating a new figure
                       fig=fig1 will plot in the given matplotlib figure
        """
        import matplotlib.pyplot as plt
        obs = self.GetPowderPatternObs()
        calc = self.GetPowderPatternCalc()

        if reset:
            self._plot_ylim = None
            self._plot_xlim = None
            self._plot_hkl = False
            self._plot_diff = False
        if 'fig' in kwargs:
            self._plot_fig = kwargs['fig']

        if diff is not None:
            self._plot_diff = diff
        plot_diff = self._plot_diff
        if hkl is not None:
            self._plot_hkl = hkl
        plot_hkl = self._plot_hkl

        # TODO: handle other coordinates than angles (TOF)
        x = np.rad2deg(self.GetPowderPatternX())
        first_plot = False
        if 'inline' not in plt.get_backend():
            if self._plot_fig is None:
                self._plot_fig = plt.figure(figsize=figsize)
            elif plt.fignum_exists(self._plot_fig.number) is False:
                self._plot_fig = plt.figure(figsize=figsize)
            plt.figure(self._plot_fig.number)
            plt.clf()
        else:
            plt.figure(figsize=figsize)
        plt.plot(x, obs, 'k', label='obs', linewidth=1)
        plt.plot(x, calc, 'r', label='calc', linewidth=1)
        if plot_diff:
            plt.plot(x, calc - obs - obs.max() / 20, 'g', label='calc-obs',
                     linewidth=0.5)
        plt.legend(loc='upper right')
        plt.title("PowderPattern: %s" % self.GetName())

        m = self.GetMaxSinThetaOvLambda() * self.GetWavelength()
        if self._plot_ylim is not None:
            plt.ylim(self._plot_ylim)
        if self._plot_xlim is not None:
            plt.xlim(self._plot_xlim)
        elif m < 1:
            plt.xlim(x.min(), np.rad2deg(np.arcsin(m)) * 2)

        if plot_hkl:
            # Plot a maximum number of hkl reflections
            nb_max = 100
            for ic in range(self.GetNbPowderPatternComponent()):
                c = self.GetPowderPatternComponent(ic)
                if isinstance(c, PowderPatternDiffraction) is False:
                    continue
                # print("HKL for:", c.GetName())
                xmin, xmax = plt.xlim()
                vh = np.round(c.GetH()).astype(np.int16)
                vk = np.round(c.GetK()).astype(np.int16)
                vl = np.round(c.GetL()).astype(np.int16)
                stol = c.GetSinThetaOverLambda()

                if 'inline' not in plt.get_backend():
                    # 'inline' backend triggers a delayed exception (?)
                    try:
                        # need the renderer to avoid text overlap
                        renderer = plt.gcf().canvas.renderer
                    except:
                        # Force immediate display. Not supported on all backends (e.g. nbagg)
                        plt.draw()
                        plt.pause(.001)
                        try:
                            renderer = plt.gcf().canvas.renderer
                        except:
                            renderer = None
                else:
                    renderer = None

                props = {'ha': 'center', 'va': 'bottom'}
                ct = 0
                last_bbox = None
                ax = plt.gca()
                tdi = ax.transData.inverted()
                for i in range(len(vh)):
                    xhkl = np.rad2deg(self.X2XCorr(self.STOL2X(stol[i])))
                    idxhkl = int(round(self.X2PixelCorr(self.STOL2X(stol[i]))))
                    # print(vh[i], vk[i], vl[i], xhkl, idxhkl)
                    if xhkl < xmin or idxhkl < 0:
                        continue
                    if xhkl > xmax or idxhkl >= len(x):
                        break
                    ct += 1
                    if ct >= nb_max:
                        break

                    ihkl = max(calc[idxhkl], obs[idxhkl])
                    s = " %d %d %d" % (vh[i], vk[i], vl[i])
                    t = plt.text(xhkl, ihkl, s, props, rotation=90,
                                 fontsize=fontsize_hkl, fontweight='light')
                    if renderer is not None:
                        # Check for overlap with previous
                        bbox = t.get_window_extent(renderer)
                        # print(s, bbox)
                        if last_bbox is not None:
                            if bbox.overlaps(last_bbox):
                                b = bbox.transformed(tdi)
                                t.set_y(ihkl + b.height * 1.2)
                        last_bbox = t.get_window_extent(renderer)
        if 'inline' not in plt.get_backend():
            try:
                # Force immediate display. Not supported on all backends (e.g. nbagg)
                plt.draw()
                plt.pause(.001)
            except:
                pass
            self._plot_xlim_plot = plt.xlim()
            plt.gca().callbacks.connect('xlim_changed', self._on_xlims_change)
            plt.gca().callbacks.connect('ylim_changed', self._on_ylims_change)
            self._plot_fig.canvas.mpl_connect('button_press_event', self._on_mouse_event)

    def quick_fit_profile(self, pdiff=None, auto_background=True, init_profile=True, plot=True,
                          zero=True, constant_width=True, width=True, eta=True, backgd=True, cell=True,
                          anisotropic=False, asym=False, displ_transl=False, verbose=True):
        if plot:
            self.plot()
        if auto_background:
            # Add background if necessary
            need_background = True
            for i in range(self.GetNbPowderPatternComponent()):
                if isinstance(self.GetPowderPatternComponent(i), PowderPatternBackground):
                    need_background = False
                    break
            if need_background:
                if verbose:
                    print("No background, adding one automatically")
                x = self.GetPowderPatternX()
                bx = np.linspace(x.min(), x.max(), 20)
                by = np.zeros(bx.shape)
                b = self.AddPowderPatternBackground()
                b.SetInterpPoints(bx, by)
                # b.Print()
                b.UnFixAllPar()
                b.OptimizeBayesianBackground()
        if pdiff is None:
            # Probably just one diffraction phase, select it
            for i in range(self.GetNbPowderPatternComponent()):
                if isinstance(self.GetPowderPatternComponent(i), PowderPatternDiffraction):
                    pdiff = self.GetPowderPatternComponent(i)
                    break
            if verbose:
                print("Selected PowderPatternDiffraction: ", pdiff.GetName(),
                      " with Crystal: ", pdiff.GetCrystal().GetName())
        if init_profile:
            pdiff.SetReflectionProfilePar(ReflectionProfileType.PROFILE_PSEUDO_VOIGT, 0.0000001)

        pdiff.SetExtractionMode(True, True)
        pdiff.ExtractLeBail(10)

        if plot:
            self.UpdateDisplay()
        # LSQ
        lsq = LSQ()
        lsq.SetRefinedObj(self, 0, True, True)
        lsq.PrepareRefParList(True)
        # lsq.GetCompiledRefinedObj().Print()
        lsqr = lsq.GetCompiledRefinedObj()

        lsqr.FixAllPar()
        # lsqr.Print()
        # print(lsq.ChiSquare())
        if zero:
            lsq.SetParIsFixed("Zero", False)
        if displ_transl:
            lsq.SetParIsFixed("2ThetaDispl", True)
            lsq.SetParIsFixed("2ThetaTransp", True)
        if lsqr.GetNbParNotFixed():
            lsq.SafeRefine(nbCycle=10, useLevenbergMarquardt=True, silent=True)
            pdiff.ExtractLeBail(10)
            lsq.SafeRefine(nbCycle=10, useLevenbergMarquardt=True, silent=True)
            if plot:
                self.UpdateDisplay()
        if cell:
            lsq.SetParIsFixed("a", False)
            lsq.SetParIsFixed("b", False)
            lsq.SetParIsFixed("c", False)
            lsq.SetParIsFixed("alpha", False)
            lsq.SetParIsFixed("beta", False)
            lsq.SetParIsFixed("gamma", False)
        if constant_width:
            lsq.SetParIsFixed("W", False)
        if lsqr.GetNbParNotFixed():
            lsq.SafeRefine(nbCycle=10, useLevenbergMarquardt=True, silent=True)
            pdiff.ExtractLeBail(10)
            lsq.SafeRefine(nbCycle=10, useLevenbergMarquardt=True, silent=True)
            if plot:
                self.UpdateDisplay()

        if width:
            lsq.SetParIsFixed("U", False)
            lsq.SetParIsFixed("V", False)
        if lsqr.GetNbParNotFixed():
            lsq.SafeRefine(nbCycle=10, useLevenbergMarquardt=True, silent=True)
            pdiff.ExtractLeBail(10)
            lsq.SafeRefine(nbCycle=10, useLevenbergMarquardt=True, silent=True)
            if plot:
                self.UpdateDisplay()

        if eta:
            lsq.SetParIsFixed("Eta0", False)
        if lsqr.GetNbParNotFixed():
            lsq.SafeRefine(nbCycle=10, useLevenbergMarquardt=True, silent=True)
            pdiff.ExtractLeBail(10)
            lsq.SafeRefine(nbCycle=10, useLevenbergMarquardt=True, silent=True)
            if plot:
                self.UpdateDisplay()

        if eta:
            lsq.SetParIsFixed("Eta1", False)
        if lsqr.GetNbParNotFixed():
            lsq.SafeRefine(nbCycle=10, useLevenbergMarquardt=True, silent=True)
            pdiff.ExtractLeBail(10)
            lsq.SafeRefine(nbCycle=10, useLevenbergMarquardt=True, silent=True)
            if plot:
                self.UpdateDisplay()

        if asym:
            lsq.SetParIsFixed("Asym0", False)
            lsq.SetParIsFixed("Asym1", False)
        if lsqr.GetNbParNotFixed():
            lsq.SafeRefine(nbCycle=10, useLevenbergMarquardt=True, silent=True)
            pdiff.ExtractLeBail(10)
            lsq.SafeRefine(nbCycle=10, useLevenbergMarquardt=True, silent=True)
            if plot:
                self.UpdateDisplay()
        if backgd:
            for i in range(self.GetNbPowderPatternComponent()):
                if isinstance(self.GetPowderPatternComponent(i), PowderPatternBackground):
                    b = self.GetPowderPatternComponent(i)
                    lsq.SetParIsFixed(refpartype_scattdata_background, False)
                    b.FixParametersBeyondMaxresolution(lsqr)
                    # lsqr.Print()
                    lsq.SafeRefine(nbCycle=10, useLevenbergMarquardt=True, silent=True)
                    break

    def _on_xlims_change(self, event_ax):
        self._plot_xlim = event_ax.get_xlim()
        if self._plot_hkl and self._plot_xlim_plot is not None:
            import matplotlib.pyplot as plt
            # Redraw to update the displayed hkl ?
            dx1 = abs(self._plot_xlim_plot[0] - plt.xlim()[0])
            dx2 = abs(self._plot_xlim_plot[1] - plt.xlim()[1])
            if max(dx1, dx2) > 0.1 * (self._plot_xlim_plot[1] - self._plot_xlim_plot[0]):
                self.plot()

    def _on_ylims_change(self, event_ax):
        self._plot_ylim = event_ax.get_ylim()

    def _on_mouse_event(self, event):
        if event.dblclick:
            self._plot_xlim = None
            self._plot_ylim = None
            self.plot()


def create_powderpattern_from_cif(file):
    """
    Create a crystal object from a CIF file or URL
    Example from Acta Cryst. (2012). B68, 407-411 (https://doi.org/10.1107/S0108768112019994),
    data hosted at CCDC:
        c=create_crystal_from_cif('https://doi.org/10.1107/S0108768112019994/ps5016sup1.cif')
        p=create_powderpattern_from_cif('https://doi.org/10.1107/S0108768112019994/ps5016Isup3.rtv')

    :param file: the filename/URL or python file object (need to open with mode='rb')
                 If the string begins by 'http' it is assumed that it is an URL instead,
                 e.g. from the crystallography open database
    :return: the imported PowderPattern
    :raises: ObjCrystException - if no PowderPattern object can be imported
    """
    p = PowderPattern()
    if isinstance(file, str):
        if len(file) > 4:
            if file[:4].lower() == 'http':
                return CreatePowderPatternFromCIF_orig(urllib.request.urlopen(file), p)
        with open(file, 'rb') as cif:  # Make sure file object is closed
            return CreatePowderPatternFromCIF_orig(cif, p)
    return CreatePowderPatternFromCIF_orig(file, p)


# PEP8
CreatePowderPatternFromCIF = create_powderpattern_from_cif
