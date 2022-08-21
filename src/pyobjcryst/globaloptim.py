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

"""Python wrapping of GlobalOptimObj.h

See the online ObjCryst++ documentation (http://vincefn.net/ObjCryst/).

Changes from ObjCryst::MonteCarloObj::
        In development !

"""
__all__ = ["MonteCarlo", "AnnealingSchedule", "GlobalOptimType"]

import warnings
from types import MethodType

try:
    import ipywidgets as widgets
except ImportError:
    widgets = None
from pyobjcryst._pyobjcryst import MonteCarlo as MonteCarlo_orig, AnnealingSchedule, \
    GlobalOptimType, OptimizationObjRegistry
from .refinableobj import *


class MonteCarlo(MonteCarlo_orig):

    def Optimize(self, nb_step: int, final_cost=0, max_time=-1):
        self._fix_parameters_for_global_optim()
        if type(self) == MonteCarlo_orig:
            self._Optimize(int(nb_step), True, final_cost, max_time)
        else:
            super().Optimize(int(nb_step), True, final_cost, max_time)

    def MultiRunOptimize(self, nb_run: int, nb_step: int, final_cost=0, max_time=-1):
        self._fix_parameters_for_global_optim()
        if type(self) == MonteCarlo_orig:
            self._MultiRunOptimize(int(nb_run), int(nb_step), True, final_cost, max_time)
        else:
            super().MultiRunOptimize(int(nb_run), int(nb_step), True, final_cost, max_time)

    def RunSimulatedAnnealing(self, nb_step: int, final_cost=0, max_time=-1):
        self._fix_parameters_for_global_optim()
        if type(self) == MonteCarlo_orig:
            self._RunSimulatedAnnealing(int(nb_step), True, final_cost, max_time)
        else:
            super().RunSimulatedAnnealing(int(nb_step), True, final_cost, max_time)

    def RunParallelTempering(self, nb_step: int, final_cost=0, max_time=-1):
        self._fix_parameters_for_global_optim()
        if type(self) == MonteCarlo_orig:
            self._RunParallelTempering(int(nb_step), True, final_cost, max_time)
        else:
            super().RunParallelTempering(int(nb_step), True, final_cost, max_time)

    def _fix_parameters_for_global_optim(self):
        # Fix parameters that should not be optimised in a MonterCarlo run
        self.SetParIsFixed(refpartype_unitcell, True)
        self.SetParIsFixed(refpartype_scattdata_scale, True)
        self.SetParIsFixed(refpartype_scattdata_profile, True)
        self.SetParIsFixed(refpartype_scattdata_corr, True)
        self.SetParIsFixed(refpartype_scattdata_background, True)
        self.SetParIsFixed(refpartype_scattdata_radiation, True)

    def widget(self):
        """
        Display a simple widget for this MonteCarlo, which only updates the current
        cost (log-likelihood). Requires ipywidgets
        """
        if widgets is None:
            warnings.warn("You need to install ipywidgets to use MonteCarlo.widget()")
            return
        self._widget = widgets.Box()
        # See https://ipywidgets.readthedocs.io/en/latest/examples/Widget%20Styling.html
        self._widget_label = widgets.Label("", layout=widgets.Layout(max_width='25%', width='20em'))
        self._widget_llk = widgets.Text("", disabled=True, layout=widgets.Layout(max_width='50%', width='30em'))
        self._widget.children = [widgets.HBox([self._widget_label, self._widget_llk])]
        self._widget_update()
        return self._widget

    def UpdateDisplay(self):
        try:
            if self._display_update_disabled:
                return
        except:
            pass
        try:
            if self._widget is not None:
                self._widget_update()
        except AttributeError:
            # self._3d_widget does not exist
            pass

    def disable_display_update(self):
        """ Disable display (useful for multiprocessing)"""
        self._display_update_disabled = True

    def enable_display_update(self):
        """ Enable display"""
        self._display_update_disabled = False

    def _widget_update(self):
        self._widget_label.value = "MonteCarlo:%s" % self.GetName()
        self._widget_label.layout.width = '%dem' % len(self._widget_label.value)
        if self.IsOptimizing():
            self._widget_llk.value = "LLK=%12.2f  Run %2d  Trial %8d" % (self.llk, self.run, self.trial)
        else:
            self._widget_llk.value = "LLK=%12.2f                        " % self.llk
        self._widget_llk.layout.width = '%dem' % len(self._widget_llk.value)


def wrap_boost_montecarlo(c: MonteCarlo):
    """
    This function is used to wrap a C++ Object by adding the python methods to it.

    :param c: the C++ created object to which the python function must be added.
    """
    if 'widget' not in dir(c):
        for func in ['Optimize', 'MultiRunOptimize', 'RunSimulatedAnnealing', 'RunParallelTempering']:
            # We keep access to the original functions... Yes, it's a kludge...
            exec("c._%s = c.%s" % (func, func))
        for func in ['Optimize', 'MultiRunOptimize', 'RunSimulatedAnnealing', 'RunParallelTempering',
                     '_fix_parameters_for_global_optim', 'widget', 'UpdateDisplay',
                     'disable_display_update', 'enable_display_update', '_widget_update']:
            exec("c.%s = MethodType(MonteCarlo.%s, c)" % (func, func))


class OptimizationObjRegistryWrapper(OptimizationObjRegistry):
    """
    Wrapper class with a GetObj() method which can correctly wrap C++ objects with
    the python methods. This is only needed when the objects have been created
    from C++, e.g. when loading an XML file.
    """

    def GetObj(self, i):
        o = self._GetObj(i)
        # TODO
        print("Casting OptimizationObj to MonteCarlo and wrapping..")
        # We get the object as an OptimizationObj, which prevents access to some functions
        # So we use this function to cast to a MonteCarloObj
        o = self._getObjCastMonteCarlo(i)
        wrap_boost_montecarlo(o)
        return o


def wrap_boost_optimizationobjregistry(o):
    """
    This function is used to wrap a C++ Object by adding the python methods to it.

    :param c: the C++ created object to which the python function must be added.
    """
    # TODO: moving the original function is not very pretty. Is there a better way ?
    if '_GetObj' not in dir(o):
        o._GetObj = o.GetObj
        o.GetObj = MethodType(OptimizationObjRegistryWrapper.GetObj, o)
