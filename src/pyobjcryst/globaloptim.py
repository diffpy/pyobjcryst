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
        self.SetParIsFixed(refpartype_unitcell, True);
        self.SetParIsFixed(refpartype_scattdata_scale, True);
        self.SetParIsFixed(refpartype_scattdata_profile, True);
        self.SetParIsFixed(refpartype_scattdata_corr, True);
        self.SetParIsFixed(refpartype_scattdata_background, True);
        self.SetParIsFixed(refpartype_scattdata_radiation, True);


    def UpdateDisplay(self):
        if self.IsOptimizing():
            print("Run %2d  Trial %8d  LLK=%12.2f" % (self.run, self.trial, self.llk))
        else:
            print("MonteCarlo: current LLK=%12.2f" % self.llk)
