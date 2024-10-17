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

See the online ObjCryst++ documentation (https://objcryst.readthedocs.io/en/latest/).

Changes from ObjCryst::PowderPattern::
        Additional functions for plotting, basic QPA and profile fitting.
"""

from urllib.request import urlopen
from packaging.version import parse as version_parse
from multiprocessing import current_process
import numpy as np

__all__ = ["PowderPattern", "CreatePowderPatternFromCIF",
           "PowderPatternBackground", "PowderPatternComponent",
           "PowderPatternDiffraction", "ReflectionProfileType",
           "SpaceGroupExplorer"]

from types import MethodType
from pyobjcryst._pyobjcryst import PowderPattern as PowderPattern_objcryst
from pyobjcryst._pyobjcryst import CreatePowderPatternFromCIF as CreatePowderPatternFromCIF_orig
from pyobjcryst._pyobjcryst import PowderPatternBackground
from pyobjcryst._pyobjcryst import PowderPatternComponent
from pyobjcryst._pyobjcryst import PowderPatternDiffraction
from pyobjcryst._pyobjcryst import ReflectionProfileType
from pyobjcryst._pyobjcryst import LSQ
from pyobjcryst.refinableobj import refpartype_scattdata_background
from pyobjcryst._pyobjcryst import SpaceGroupExplorer
from pyobjcryst import ObjCrystException


class PowderPattern(PowderPattern_objcryst):

    def __init__(self):
        super(PowderPattern, self).__init__()
        self._plot_fig = None
        self._plot_xlim = None
        self._plot_ylim = None
        self._plot_diff = False
        self._plot_hkl = False
        self._plot_hkl_fontsize = 6
        self._plot_phase_labels = None
        # xlim last time hkl were plotted
        self._last_hkl_plot_xlim = None
        self.evts = []
        self._colour_phases = ["black", "blue", "green", "red", "brown", "olive",
                               "cyan", "purple", "magenta", "salmon"]

    def UpdateDisplay(self):
        try:
            if self._display_update_disabled:
                return
        except:
            pass
        if self._plot_fig is not None:
            if self._plot_fig is not None:
                self.plot()

    def update_display(self, return_figure=False):
        """
        Update the plotted figure (if it exists)

        :param return_figure: if True, returns the figure
        :return: the figure if return_figure is True
        """
        self.UpdateDisplay()
        if return_figure:
            return self.figure

    def disable_display_update(self):
        """ Disable display (useful for multiprocessing)"""
        self._display_update_disabled = True

    def enable_display_update(self):
        """ Enable display"""
        self._display_update_disabled = False

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
        try:
            calc = self.GetPowderPatternCalc()
        except ObjCrystException:
            # TODO: when importing  objects from an XML file, the powder pattern does not compute
            #  correctly, Prepare() needs to be called manually. Why ?
            self.Prepare()
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
        if self._plot_fig is None or 'inline' in plt.get_backend():
            self._plot_fig = plt.figure(figsize=figsize)
        else:
            self._plot_fig.clear()
        ax = self._plot_fig.axes[0] if len(self._plot_fig.axes) else self._plot_fig.subplots()
        ax.plot(x, obs, 'k', label='obs', linewidth=1)
        ax.plot(x, calc, 'r', label='calc', linewidth=1)
        m = min(1, self.GetMaxSinThetaOvLambda() * self.GetWavelength())
        mtth = np.rad2deg(np.arcsin(m)) * 2
        if plot_diff:
            diff = calc - obs - obs.max() / 20
            # Mask difference above max sin(theta)/lambda
            diff = np.ma.masked_array(diff, x > mtth)
            ax.plot(x, diff, 'g', label='calc-obs',
                    linewidth=0.5)

        ax.legend(loc='upper right')
        if self.GetName() != "":
            self._plot_fig.title("PowderPattern: %s" % self.GetName())

        if self._plot_ylim is not None:
            ax.set_ylim(self._plot_ylim)
        if self._plot_xlim is not None:
            ax.set_xlim(self._plot_xlim)
        elif m < 1:
            ax.set_xlim(x.min(), mtth)

        if plot_hkl:
            self._do_plot_hkl(nb_max=100, fontsize_hkl=fontsize_hkl)

        # print PowderPatternDiffraction names
        self._plot_phase_labels = []
        iphase = 0
        for i in range(self.GetNbPowderPatternComponent()):
            s = ""
            comp = self.GetPowderPatternComponent(i)
            if comp.GetClassName() == "PowderPatternDiffraction":
                if comp.GetName() != "":
                    s += "%s\n" % comp.GetName()
                else:
                    c = comp.GetCrystal()
                    if c.GetName() != "":
                        s += c.GetName()
                    else:
                        s += c.GetFormula()
                    s += "[%s]" % str(c.GetSpaceGroup())
                if comp.GetExtractionMode():
                    s += "[Le Bail mode]"
                self._plot_phase_labels.append(s)
                ax.text(0.005, 0.995, "\n" * iphase + s, horizontalalignment="left", verticalalignment="top",
                        transform=ax.transAxes, fontsize=6, color=self._colour_phases[iphase])
                iphase += 1

        if 'inline' not in plt.get_backend():
            try:
                # Force immediate display. Not supported on all backends (e.g. nbagg)
                ax.draw()
                self._plot_fig.canvas.draw()
                if 'ipympl' not in plt.get_backend():
                    plt.pause(.001)
            except:
                pass
            # plt.gca().callbacks.connect('xlim_changed', self._on_xlims_change)
            # plt.gca().callbacks.connect('ylim_changed', self._on_ylims_change)
            self._plot_fig.canvas.mpl_connect('button_press_event', self._on_mouse_event)
            self._plot_fig.canvas.mpl_connect('draw_event', self._on_draw_event)

    def _do_plot_hkl(self, nb_max=100, fontsize_hkl=None):
        import matplotlib.pyplot as plt
        from matplotlib import __version__ as mpl_version
        if fontsize_hkl is None:
            fontsize_hkl = self._plot_hkl_fontsize
        else:
            self._plot_hkl_fontsize = fontsize_hkl
        ax = self._plot_fig.axes[0]
        # Plot up to nb_max hkl reflections
        obs = self.GetPowderPatternObs()
        calc = self.GetPowderPatternCalc()
        x = np.rad2deg(self.GetPowderPatternX())
        # Clear previous text (assumes only hkl were printed)
        if version_parse(mpl_version) < version_parse("3.7.0"):
            # This will fail for matplotlib>=(3,7,0)
            ax.texts.clear()
        else:
            for t in ax.texts:
                t.remove()
        iphase = 0
        for ic in range(self.GetNbPowderPatternComponent()):
            c = self.GetPowderPatternComponent(ic)
            if isinstance(c, PowderPatternDiffraction) is False:
                continue
            # print("HKL for:", c.GetName())
            xmin, xmax = ax.get_xlim()
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
                    ax.draw()
                    self._plot_fig.canvas.draw()
                    if 'ipympl' not in plt.get_backend():
                        plt.pause(.001)
                    try:
                        renderer = self._plot_fig.canvas.renderer
                    except:
                        renderer = None
            else:
                renderer = None

            props = {'ha': 'center', 'va': 'bottom'}
            ct = 0
            last_bbox = None
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
                t = ax.text(xhkl, ihkl, s, props, rotation=90, fontsize=fontsize_hkl,
                            fontweight='light', color=self._colour_phases[iphase])
                if renderer is not None:
                    # Check for overlap with previous
                    bbox = t.get_window_extent(renderer)
                    # print(s, bbox)
                    if last_bbox is not None:
                        if bbox.overlaps(last_bbox):
                            b = bbox.transformed(tdi)
                            t.set_y(ihkl + b.height * 1.2)
                    last_bbox = t.get_window_extent(renderer)
            iphase += 1
        self._last_hkl_plot_xlim = ax.get_xlim()
        if self._plot_phase_labels is not None:
            for iphase in range(len(self._plot_phase_labels)):
                s = self._plot_phase_labels[iphase]
                ax.text(0.005, 0.995, "\n" * iphase + s, horizontalalignment="left", verticalalignment="top",
                        transform=ax.transAxes, fontsize=6, color=self._colour_phases[iphase])

    @property
    def figure(self):
        """
        return: the figure used for plotting, or None. Note that
            if you want to display it in a notebook using ipympl (aka
            'matplotlib widget'), you should 'figure.canvas' to display
            also the toolbar (zoom, etc...).
        """
        return self._plot_fig

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
        if verbose:
            print("Profile fitting finished.\n"
                  "Remember to use SetExtractionMode(False) on the PowderPatternDiffraction object\n"
                  "to disable profile fitting and optimise the structure.")

    def get_background(self):
        """
        Access the background component.

        :return: the PowderPatternBackground for this powder pattern, or None
        """
        for i in range(self.GetNbPowderPatternComponent()):
            if self.GetPowderPatternComponent(i).GetClassName() == "PowderPatternBackground":
                return self.GetPowderPatternComponent(i)

    def get_crystalline_components(self):
        """
        Get the crystalline phase for this powder pattern
        :return: a list of the PowderPatternDiffraction components
        """
        vc = []
        for i in range(self.GetNbPowderPatternComponent()):
            if self.GetPowderPatternComponent(i).GetClassName() == "PowderPatternDiffraction":
                vc.append(self.GetPowderPatternComponent(i))
        return vc

    def _on_mouse_event(self, event):
        if event.dblclick:
            # This does not work in a notebook
            self._plot_xlim = None
            self._plot_ylim = None
            self.plot()

    def _on_draw_event(self, event):
        if self._plot_hkl and self._last_hkl_plot_xlim is not None and len(self._plot_fig.axes):
            ax = self._plot_fig.axes[0]
            self._plot_xlim = ax.get_xlim()
            dx1 = abs(self._last_hkl_plot_xlim[0] - self._plot_xlim[0])
            dx2 = abs(self._last_hkl_plot_xlim[1] - self._plot_xlim[1])
            if max(dx1, dx2) > 0.1 * (self._last_hkl_plot_xlim[1] - self._last_hkl_plot_xlim[0]):
                # Need to update the hkl list
                self._do_plot_hkl()

    def qpa(self, verbose=False):
        """
        Get the quantitative phase analysis for the current powder pattern,
        when multiple crystalline phases are present.

        :param verbose: if True, print the Crystal names and their weight percentage.
        :return: a dictionary with the PowderPatternDiffraction object as key, and
            the weight percentages as value.
        """
        res = {}
        szmv_sum = 0
        for pdiff in self.get_crystalline_components():
            if not isinstance(pdiff, PowderPatternDiffraction):
                continue
            c = pdiff.GetCrystal()
            s = self.GetScaleFactor(pdiff)
            m = c.GetWeight()
            z = c.GetSpaceGroup().GetNbSymmetrics()
            v = c.GetVolume()
            # print("%25s: %12f, %10f, %3d, %10.2f" % (c.GetName(), s, m, z, v))
            res[pdiff] = s * z * m * v
            szmv_sum += s * z * m * v

        if verbose:
            print("Weight percentages:")
        for k, v in res.items():
            res[k] = v / szmv_sum
            if verbose:
                print("%25s: %6.2f%%" % (k.GetCrystal().GetName(), res[k] * 100))
        return res


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
                return CreatePowderPatternFromCIF_orig(urlopen(file), p)
        with open(file, 'rb') as cif:  # Make sure file object is closed
            return CreatePowderPatternFromCIF_orig(cif, p)
    return CreatePowderPatternFromCIF_orig(file, p)


def wrap_boost_powderpattern(c: PowderPattern):
    """
    This function is used to wrap a C++ Object by adding the python methods to it.

    :param c: the C++ created object to which the python function must be added.
    """
    if '_plot_fig' not in dir(c):
        # Add attributes
        c._plot_fig = None
        c._plot_xlim = None
        c._plot_ylim = None
        c._plot_diff = False
        c._plot_hkl = False
        c._plot_hkl_fontsize = 6
        c._plot_phase_labels = None
        c._last_hkl_plot_xlim = None
        c.evts = []
        c._colour_phases = ["black", "blue", "green", "red", "brown", "olive",
                            "cyan", "purple", "magenta", "salmon"]
        for func in ['UpdateDisplay', 'disable_display_update', 'enable_display_update', 'plot',
                     '_do_plot_hkl', 'quick_fit_profile', 'get_background', 'get_crystalline_components',
                     '_on_mouse_event', '_on_draw_event', 'qpa']:
            exec("c.%s = MethodType(PowderPattern.%s, c)" % (func, func))


# PEP8
CreatePowderPatternFromCIF = create_powderpattern_from_cif
