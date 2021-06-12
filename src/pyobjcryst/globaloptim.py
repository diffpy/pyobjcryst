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
try:
    import ipywidgets as widgets
except ImportError:
    widgets = None
from pyobjcryst._pyobjcryst import MonteCarlo as MonteCarlo_orig, AnnealingSchedule, GlobalOptimType
from .refinableobj import *


class MonteCarlo(MonteCarlo_orig):

    def Optimize(self, nb_step: int, final_cost=0, max_time=-1):
        self._fix_parameters_for_global_optim()
        super().Optimize(int(nb_step), True, final_cost, max_time)

    def MultiRunOptimize(self, nb_run: int, nb_step: int, final_cost=0, max_time=-1):
        self._fix_parameters_for_global_optim()
        super().MultiRunOptimize(int(nb_run), int(nb_step), True, final_cost, max_time)

    def RunSimulatedAnnealing(self, nb_step: int, final_cost=0, max_time=-1):
        self._fix_parameters_for_global_optim()
        super().RunSimulatedAnnealing(int(nb_step), True, final_cost, max_time)

    def RunParallelTempering(self, nb_step: int, final_cost=0, max_time=-1):
        self._fix_parameters_for_global_optim()
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
        Display a simple widget for this MonteCarloObj, which only updates the current
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
