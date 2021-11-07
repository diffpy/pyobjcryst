#!/usr/bin/env python
##############################################################################
#
# pyobjcryst        by DANSE Diffraction group
#                   Simon J. L. Billinge
#                   (c) 2009 The Trustees of Columbia University
#                   in the City of New York.  All rights reserved.
#
# File coded by:    Chris Farrow, Vincent Favre-Nicolin
#
# See AUTHORS.txt for a list of people who contributed.
# See LICENSE_DANSE.txt for license information.
#
##############################################################################

"""Python wrapping of Crystal.h.

See the online ObjCryst++ documentation (http://vincefn.net/ObjCryst/).

Changes from ObjCryst::Crystal

- CIFOutput accepts a python file-like object
- CalcDynPopCorr is not enabled, as the API states that this is for internal
  use only.

Other Changes

- CreateCrystalFromCIF is placed here instead of in a seperate CIF module. This
  method accepts a python file or a filename rather than a CIF object.
"""

__all__ = ["Crystal", "BumpMergePar", "CreateCrystalFromCIF",
           "create_crystal_from_cif", "gCrystalRegistry"]

import warnings
from urllib.request import urlopen
from multiprocessing import current_process
import numpy as np
from pyobjcryst._pyobjcryst import Crystal as Crystal_orig
from pyobjcryst._pyobjcryst import BumpMergePar
from pyobjcryst._pyobjcryst import CreateCrystalFromCIF as CreateCrystalFromCIF_orig
from pyobjcryst._pyobjcryst import gCrystalRegistry

try:
    import py3Dmol
except ImportError:
    py3Dmol = None

try:
    import ipywidgets as widgets
except ImportError:
    widgets = None


class Crystal(Crystal_orig):

    def CIFOutput(self, file, mindist=0):
        """
        Save the crystal structure to a CIF file.

        :param file: either a filename, or a python file object opened in write mode
        """
        if isinstance(file, str):
            super().CIFOutput(open(file, "w"), mindist)
        else:
            super().CIFOutput(file, mindist)

    def UpdateDisplay(self):
        try:
            if self._display_update_disabled:
                return
        except:
            pass
        # test for _3d_widget is a bit ugly, but to correctly implement this we'd need an
        # __init__ function which overrides the 3 different Crystal constructors which
        # could be messy as well.
        try:
            if self._3d_widget is not None:
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

    def _display_cif(self, xmin=0, xmax=1, ymin=0, ymax=1, zmin=0, zmax=1, enantiomer=False,
                     full_molecule=True, only_independent_atoms=False):
        """
        Create a CIF with the full list of atoms, including those deduced by symmetry
        or translation up to neighbouring unit cells

        :param xmin, xmax, ymin, ymax, zmin, zmax: the view limits in fractional coordinates.
        :param enantiomer: if True, will mirror the structure along the x axis
        :param full_molecule: if True, a Molecule (or Scatterer) which has at least
            one atom inside the view limits is entirely shown.
        :param only_independent_atoms: if True, only show the independent atoms, no symmetry
            or translation is applied
        :return : the CIF as a string
        """
        cif = "data_crystal_for3d\n\n"
        cif += "_computing_structure_solution     'FOX http://objcryst.sourceforge.net'\n\n";
        cif += "_cell_length_a       %8.3f\n" % self.a
        cif += "_cell_length_b       %8.3f\n" % self.b
        cif += "_cell_length_c       %8.3f\n" % self.c
        cif += "_cell_length_alpha   %8.3f\n" % np.rad2deg(self.alpha)
        cif += "_cell_length_beta    %8.3f\n" % np.rad2deg(self.beta)
        cif += "_cell_length_gamma   %8.3f\n" % np.rad2deg(self.gamma)

        cif += "loop_\n"
        cif += "    _atom_site_label\n"
        cif += "    _atom_site_type_symbol\n"
        cif += "    _atom_site_fract_x\n"
        cif += "    _atom_site_fract_y\n"
        cif += "    _atom_site_fract_z\n"
        cif += "    _atom_site_occupancy\n"

        spg = self.GetSpaceGroup()

        for i in range(self.GetNbScatterer()):
            scatt = self.GetScatterer(i)
            v = scatt.GetScatteringComponentList()
            nat = len(v)

            if only_independent_atoms:
                for j in range(len(v)):
                    s = v[j]
                    symbol = s.mpScattPow.GetSymbol()
                    name = scatt.GetComponentName(j)
                    # 3dmol.js does not like ' in names,
                    # despite https://www.iucr.org/resources/cif/spec/version1.1/cifsyntax#bnf
                    name = name.replace("'", "_")
                    occ = s.Occupancy
                    x, y, z = s.X % 1, s.Y % 1, s.Z % 1
                    if enantiomer:
                        x = -x % 1
                    cif += "    %12s %4s %8.4f %8.4f %8.4f %6.4f\n" % (name, symbol, x, y, z, occ)
            else:
                # Generate all symmetrics to enable full molecule display
                nsym = spg.GetNbSymmetrics()
                # print(nsym)
                vxyz = np.empty((nsym, nat, 3), dtype=np.float32)
                for j in range(nat):
                    s = v[j]
                    x, y, z = s.X, s.Y, s.Z
                    if enantiomer:
                        x = -x
                    xyzsym = spg.GetAllSymmetrics(x, y, z)
                    for k in range(nsym):
                        vxyz[k, j, :] = xyzsym[k]

                for k in range(nsym):
                    xc, yc, zc = vxyz[k].mean(axis=0)
                    vxyz[k, :, 0] -= (xc - xc % 1)
                    vxyz[k, :, 1] -= (yc - yc % 1)
                    vxyz[k, :, 2] -= (zc - zc % 1)

                # print(vxyz, vxyz.shape)

                for j in range(nat):
                    s = v[j]
                    symbol = s.mpScattPow.GetSymbol()
                    name = scatt.GetComponentName(j)
                    # 3dmol.js does not like ' in names,
                    # despite https://www.iucr.org/resources/cif/spec/version1.1/cifsyntax#bnf
                    name = name.replace("'", "_")
                    occ = s.Occupancy

                    for k in range(nsym):
                        for dx in (-1, 0, 1):
                            for dy in (-1, 0, 1):
                                for dz in (-1, 0, 1):
                                    x, y, z = vxyz[k, j] + np.array((dx, dy, dz))
                                    # print("    %12s %4s %8.4f %8.4f %8.4f %6.4f" % \
                                    #           (name, symbol, x, y, z, occ))
                                    if full_molecule:
                                        # If any atom is within limits, display all
                                        vx, vy, vz = vxyz[k, :, 0] + dx, vxyz[k, :, 1] + dy, vxyz[k, :, 2] + dz
                                        tmp = (vx >= xmin) * (vx <= xmax) * (vy >= ymin) * \
                                              (vy <= ymax) * (vz >= zmin) * (vz <= zmax)
                                        if tmp.sum():
                                            cif += "    %12s %4s %8.4f %8.4f %8.4f %6.4f\n" % \
                                                   (name, symbol, x, y, z, occ)
                                    else:
                                        if xmin <= x <= xmax and ymin <= y <= ymax and zmin <= z <= zmax:
                                            cif += "    %12s %4s %8.4f %8.4f %8.4f %6.4f\n" % \
                                                   (name, symbol, x, y, z, occ)
        return cif

    def _display_list(self, xmin=0, xmax=1, ymin=0, ymax=1, zmin=0, zmax=1, enantiomer=False,
                      full_molecule=True, only_independent_atoms=False):
        """
        Create a list of atoms to be displayed, so it can be supplied to py3Dmol

        :param xmin, xmax, ymin, ymax, zmin, zmax: the view limits in fractional coordinates.
        :param enantiomer: if True, will mirror the structure along the x axis
        :param full_molecule: if True, a Molecule (or Scatterer) which has at least
            one atom inside the view limits is entirely shown.
        :param only_independent_atoms: if True, only show the independent atoms, no symmetry
            or translation is applied
        :return : the list of atoms and bonds to be displayed for 3dmol
        """

        spg = self.GetSpaceGroup()
        vv = []
        idx = 0
        for i in range(self.GetNbScatterer()):
            scatt = self.GetScatterer(i)
            v = scatt.GetScatteringComponentList()
            nat = len(v)
            if scatt.GetClassName() == "Molecule":
                # We need to generate all atomic positions and the associated bonds
                atoms = {}
                for j in range(nat):
                    s = v[j]
                    a = scatt.GetAtom(j)
                    if a.IsDummy():
                        continue
                    name = scatt.GetComponentName(j)
                    name = name.replace("'", "_")
                    symbol = s.mpScattPow.GetSymbol()
                    occ = s.Occupancy
                    x, y, z = s.X, s.Y, s.Z
                    if enantiomer:
                        x = -x
                    atoms[a.int_ptr()] = {'x': x, 'y': y, 'z': z, 'name': name, 'j': j,
                                          'symbol': symbol, 'bonds': [], 'bondOrder': []}
                for bond in scatt.IterBond():
                    o = bond.BondOrder
                    if o == 0:
                        o = 1
                    i1 = bond.GetAtom1().int_ptr()
                    i2 = bond.GetAtom2().int_ptr()
                    atoms[i1]['bonds'].append(i2)
                    atoms[i2]['bonds'].append(i1)
                    atoms[i1]['bondOrder'].append(o)
                    atoms[i2]['bondOrder'].append(o)
                if only_independent_atoms:
                    # Generate the index for the atoms
                    for a in atoms.values():
                        a['idx'] = idx
                        idx += 1
                    for a in atoms.values():
                        vb = [atoms[int_ptr]['idx'] for int_ptr in a['bonds']]
                        x, y, z = self.FractionalToOrthonormalCoords(a['x'], a['y'], a['z'])
                        vv.append({'elem': a['symbol'], 'x': x, 'y': y, 'z': z,
                                   'bonds': vb, 'bondOrder': a['bondOrder']})
                else:
                    # Generate all symmetrics to enable full molecule display
                    nsym = spg.GetNbSymmetrics()
                    # print(nsym)
                    vxyz = np.empty((nsym, nat, 3), dtype=np.float32)
                    for j in range(nat):
                        s = v[j]
                        x, y, z = s.X, s.Y, s.Z
                        if enantiomer:
                            x = -x
                        xyzsym = spg.GetAllSymmetrics(x, y, z)
                        vxyz[:, j, :] = xyzsym

                    for k in range(nsym):
                        xc, yc, zc = vxyz[k].mean(axis=0)
                        vxyz[k, :, 0] -= (xc - xc % 1)
                        vxyz[k, :, 1] -= (yc - yc % 1)
                        vxyz[k, :, 2] -= (zc - zc % 1)

                    if full_molecule:
                        for k in range(nsym):
                            for dx in (-1, 0, 1):
                                for dy in (-1, 0, 1):
                                    for dz in (-1, 0, 1):
                                        vx, vy, vz = vxyz[k, :, 0] + dx, vxyz[k, :, 1] + dy, vxyz[k, :, 2] + dz
                                        # Is at least one atom inside the limits ?
                                        tmp = (vx >= xmin) * (vx <= xmax) * (vy >= ymin) * (vy <= ymax) * (
                                                vz >= zmin) * (vz <= zmax)
                                        if tmp.sum():
                                            for a in atoms.values():
                                                a['idx'] = idx
                                                idx += 1
                                            for a in atoms.values():
                                                j = a['j']
                                                vb = [atoms[int_ptr]['idx'] for int_ptr in a['bonds']]
                                                x, y, z = vxyz[k, j] + np.array((dx, dy, dz))
                                                x, y, z = self.FractionalToOrthonormalCoords(x, y, z)
                                                vv.append({'elem': a['symbol'], 'x': x, 'y': y, 'z': z,
                                                           'bonds': vb, 'bondOrder': a['bondOrder']})
                    else:
                        # TODO add 'visible' value in dictionnary to determine which atoms are shown,
                        # then update the bond and bondOrder lists
                        for k in range(nsym):
                            for dx in (-1, 0, 1):
                                for dy in (-1, 0, 1):
                                    for dz in (-1, 0, 1):
                                        vx, vy, vz = vxyz[k, :, 0] + dx, vxyz[k, :, 1] + dy, vxyz[k, :, 2] + dz
                                        for a in atoms.values():
                                            j = a['j']
                                            x, y, z = vx[j], vy[j], vz[j]
                                            if xmin <= x <= xmax and ymin <= y <= ymax and zmin <= z <= zmax:
                                                a['idx'] = idx
                                                a['visible'] = True
                                                idx += 1
                                            else:
                                                a['visible'] = False
                                        for a in atoms.values():
                                            if not a['visible']:
                                                continue
                                            j = a['j']
                                            vb = []
                                            vo = []
                                            for l in range(len(a['bonds'])):
                                                int_ptr = a['bonds'][l]
                                                if atoms[int_ptr]['visible']:
                                                    vb.append(atoms[int_ptr]['idx'])
                                                    vo.append(a['bondOrder'][l])
                                            x, y, z = vxyz[k, j] + np.array((dx, dy, dz))
                                            x, y, z = self.FractionalToOrthonormalCoords(x, y, z)
                                            vv.append({'elem': a['symbol'], 'x': x, 'y': y, 'z': z,
                                                       'bonds': vb, 'bondOrder': vo})
            else:
                if only_independent_atoms:
                    for j in range(len(v)):
                        s = v[j]
                        symbol = s.mpScattPow.GetSymbol()
                        name = scatt.GetComponentName(j)
                        # 3dmol.js does not like ' in names,
                        # despite https://www.iucr.org/resources/cif/spec/version1.1/cifsyntax#bnf
                        name = name.replace("'", "_")
                        occ = s.Occupancy
                        x, y, z = s.X, s.Y, s.Z
                        if enantiomer:
                            x = -x
                        x, y, z = self.FractionalToOrthonormalCoords(x, y, z)
                        vv.append({'elem': symbol, 'x': x, 'y': y, 'z': z})
                else:
                    # Generate all symmetrics to enable full molecule display
                    nsym = spg.GetNbSymmetrics()
                    # print(nsym)
                    vxyz = np.empty((nsym, nat, 3), dtype=np.float32)
                    for j in range(nat):
                        s = v[j]
                        x, y, z = s.X, s.Y, s.Z
                        if enantiomer:
                            x = -x
                        xyzsym = spg.GetAllSymmetrics(x, y, z)
                        for k in range(nsym):
                            vxyz[k, j, :] = xyzsym[k]

                    for k in range(nsym):
                        xc, yc, zc = vxyz[k].mean(axis=0)
                        vxyz[k, :, 0] -= (xc - xc % 1)
                        vxyz[k, :, 1] -= (yc - yc % 1)
                        vxyz[k, :, 2] -= (zc - zc % 1)

                    # print(vxyz, vxyz.shape)

                    for j in range(nat):
                        s = v[j]
                        symbol = s.mpScattPow.GetSymbol()
                        name = scatt.GetComponentName(j)
                        # 3dmol.js does not like ' in names,
                        # despite https://www.iucr.org/resources/cif/spec/version1.1/cifsyntax#bnf
                        name = name.replace("'", "_")
                        occ = s.Occupancy

                        for k in range(nsym):
                            for dx in (-1, 0, 1):
                                for dy in (-1, 0, 1):
                                    for dz in (-1, 0, 1):
                                        x, y, z = vxyz[k, j] + np.array((dx, dy, dz))
                                        if xmin <= x <= xmax and ymin <= y <= ymax and zmin <= z <= zmax:
                                            x, y, z = self.FractionalToOrthonormalCoords(x, y, z)
                                            vv.append({'elem': symbol, 'x': x, 'y': y, 'z': z})
        return vv

    def display_3d(self, xmin=0, xmax=1, ymin=0, ymax=1, zmin=0, zmax=1, enantiomer=False,
                   full_molecule_opacity=0.5, extra_dist=2, extra_opacity=0.5):
        """
        This will return a 3D view of the Crystal structure which can be displayed
        in a notebook. This cannot be automatically updated, but will remain in the
        notebook as a static javascript object, so it can still be useful.

        :param xmin, xmax, ymin, ymax, zmin, zmax: the view limits in fractional coordinates.
        :param enantiomer: if True, will mirror the structure along the x axis
        :param full_molecule_opacity: if >0, a Molecule (or Scatterer) which has at least
            one atom inside the view limits is entirely shown, with the given opacity (0-1)
        :param extra_dist: extra distance (in Angstroms) beyond the view limits, where
            atoms & bonds are still displayed semi-transparently
        :param extra_opacity: the opacity (0-1) to display the atoms within the extra distance.
        """
        if py3Dmol is None:
            warnings.warn("Yout need to install py3Dmol>=0.9 to use Crystal.display_3d()")
            return
        v = py3Dmol.view()

        if full_molecule_opacity > 0:
            v.addModel()
            m = v.getModel()
            atoms = self._display_list(xmin, xmax, ymin, ymax, zmin, zmax, full_molecule=True,
                                       only_independent_atoms=False, enantiomer=enantiomer)
            m.addAtoms(atoms)
            m.setStyle({'stick': {'radius': 0.2, 'opacity': full_molecule_opacity},
                        'sphere': {'scale': 0.3, 'colorscheme': 'jmol', 'opacity': full_molecule_opacity}})

        if extra_opacity > 0 and extra_dist > 0:
            dx, dy, dz = extra_dist / self.a, extra_dist / self.b, extra_dist / self.c
            v.addModel()
            m = v.getModel()
            atoms = self._display_list(xmin - dx, xmax + dx, ymin - dy, ymax + dy, zmin - dz, zmax + dz,
                                       full_molecule=False,
                                       only_independent_atoms=False, enantiomer=enantiomer)
            m.addAtoms(atoms)
            m.setStyle({'stick': {'radius': 0.2, 'opacity': extra_opacity},
                        'sphere': {'scale': 0.3, 'colorscheme': 'jmol', 'opacity': extra_opacity}})

        v.addModel()
        m = v.getModel()
        m.setCrystData(self.a, self.b, self.c, np.rad2deg(self.alpha), np.rad2deg(self.beta), np.rad2deg(self.gamma))
        v.addUnitCell({'box': {'color': 'purple'}, 'alabel': 'X', 'blabel': 'Y', 'clabel': 'Z',
                       'alabelstyle': {'fontColor': 'black', 'backgroundColor': 'white', 'inFront': True,
                                       'fontSize': 40},
                       'astyle': {'color': 'darkred', 'radius': 5, 'midpos': -10}})

        atoms = self._display_list(xmin, xmax, ymin, ymax, zmin, zmax, full_molecule=False,
                                   only_independent_atoms=False, enantiomer=enantiomer)
        m.addAtoms(atoms)
        m.setStyle({'stick': {'radius': 0.2, 'opacity': 1},
                    'sphere': {'scale': 0.3, 'colorscheme': 'jmol', 'opacity': 1}})

        v.zoomTo()
        return v

    def widget_3d(self, xmin=0, xmax=1, ymin=0, ymax=1, zmin=0, zmax=1, enantiomer=False,
                  full_molecule_opacity=0.5, extra_dist=2, extra_opacity=0.5, width=640, height=480):
        """
        This will return a 3D view of the Crystal structure which can be displayed
        in a notebook, along with controls for the display. This can be live-updated.

        :param xmin, xmax, ymin, ymax, zmin, zmax: the view limits in fractional coordinates.
        :param enantiomer: if True, will mirror the structure along the x axis
        :param full_molecule_opacity: if >0, a Molecule (or Scatterer) which has at least
            one atom inside the view limits is entirely shown, with the given opacity (0-1)
        :param extra_dist: extra distance (in Angstroms) beyond the view limits, where
            atoms & bonds are still displayed semi-transparently
        :param extra_opacity: the opacity (0-1) to display the atoms within the extra distance.
        :param width, height: the width and height of the 3D view
        """
        if widgets is None or py3Dmol is None:
            warnings.warn("You need to install py3Dmol>=0.9 and ipywidgets to use Crystal.widget_3d()")
            return
        self._3d_widget = widgets.Box()

        # TODO: toggle for labels, toggle for stick (bonds), radius for atoms, enantiomer

        # Use a step of ~0.5 Angstroem
        xstep = 0.5 / self.a
        # Adapt step so we can keep orginal values as integral number steps
        xstep = (xmax - xmin) / np.ceil((xmax - xmin) / xstep)
        self.xrange = widgets.FloatRangeSlider(value=[xmin, xmax], min=xmin - 0.5, max=xmax + 0.5,
                                               step=xstep, description='Xrange',
                                               disabled=False, continuous_update=True, orientation='horizontal',
                                               readout=True)
        ystep = 0.5 / self.b
        ystep = (ymax - ymin) / np.ceil((ymax - ymin) / ystep)
        self.yrange = widgets.FloatRangeSlider(value=[ymin, ymax], min=ymin - 0.5, max=ymax + 0.5,
                                               step=ystep, description='Yrange',
                                               disabled=False, continuous_update=True, orientation='horizontal',
                                               readout=True)

        zstep = 0.5 / self.c
        zstep = (zmax - zmin) / np.ceil((zmax - zmin) / zstep)
        self.zrange = widgets.FloatRangeSlider(value=[zmin, zmax], min=zmin - 0.5, max=zmax + 0.5,
                                               step=zstep, description='Zrange',
                                               disabled=False, continuous_update=True, orientation='horizontal',
                                               readout=True)

        self.vbox_range = widgets.VBox([self.xrange, self.yrange, self.zrange])

        self.extra_dist = widgets.FloatSlider(value=extra_dist, min=0, max=10, step=0.5,
                                              description='extra dist',
                                              tooltip='Extra distance (A) with semi-transparent display',
                                              disabled=False, continuous_update=True, orientation='horizontal',
                                              readout=True, readout_format='.1f')

        self.extra_opacity = widgets.FloatSlider(value=extra_opacity, min=0, max=1, step=0.1,
                                                 description='extra opac.',
                                                 tooltip='Opacity for extra distance display',
                                                 disabled=False, continuous_update=True, orientation='horizontal',
                                                 readout=True, readout_format='.01f')

        self.full_molecule_opacity = widgets.FloatSlider(value=full_molecule_opacity, min=0, max=1, step=0.1,
                                                         description='fullMol opac',
                                                         tooltip='Opacity to display fully molecules\n'
                                                                 'which have at least one atom inside the limits',
                                                         disabled=False, continuous_update=True,
                                                         orientation='horizontal',
                                                         readout=True, readout_format='.01f')

        self.vbox_options = widgets.VBox([self.extra_dist, self.extra_opacity, self.full_molecule_opacity])

        self.hbox_options = widgets.HBox([self.vbox_range, self.vbox_options])

        # catch the py3dmol display in widgets.Output ?
        self.output_view = widgets.Output()
        with self.output_view:
            self.py3dmol_view = py3Dmol.view(width=width, height=height)

        self.vbox = widgets.VBox([self.hbox_options, self.output_view])
        self._3d_widget.children = [self.vbox]

        self._widget_update(show=True, zoom=True)

        self.xrange.observe(self._widget_on_change_parameter)
        self.yrange.observe(self._widget_on_change_parameter)
        self.zrange.observe(self._widget_on_change_parameter)
        self.extra_dist.observe(self._widget_on_change_parameter)
        self.extra_opacity.observe(self._widget_on_change_parameter)
        self.full_molecule_opacity.observe(self._widget_on_change_parameter)

        return self._3d_widget

    def _widget_update(self, show=False, zoom=False):
        xmin, xmax = self.xrange.value
        ymin, ymax = self.yrange.value
        zmin, zmax = self.zrange.value
        extra_dist = self.extra_dist.value
        extra_opacity = self.extra_opacity.value
        full_molecule_opacity = self.full_molecule_opacity.value
        v = self.py3dmol_view
        v.removeAllModels()
        if full_molecule_opacity > 0:
            v.addModel()
            m = v.getModel()
            atoms = self._display_list(xmin, xmax, ymin, ymax, zmin, zmax, full_molecule=True,
                                       only_independent_atoms=False)
            m.addAtoms(atoms)
            m.setStyle({'stick': {'radius': 0.2, 'opacity': full_molecule_opacity},
                        'sphere': {'scale': 0.3, 'colorscheme': 'jmol', 'opacity': full_molecule_opacity}})

        if extra_opacity > 0 and extra_dist > 0:
            dx, dy, dz = extra_dist / self.a, extra_dist / self.b, extra_dist / self.c
            v.addModel()
            m = v.getModel()
            atoms = self._display_list(xmin - dx, xmax + dx, ymin - dy, ymax + dy, zmin - dz, zmax + dz,
                                       full_molecule=False, only_independent_atoms=False)
            m.addAtoms(atoms)
            m.setStyle({'stick': {'radius': 0.2, 'opacity': extra_opacity},
                        'sphere': {'scale': 0.3, 'colorscheme': 'jmol', 'opacity': extra_opacity}})

        v.addModel()
        m = v.getModel()
        m.setCrystData(self.a, self.b, self.c, np.rad2deg(self.alpha), np.rad2deg(self.beta), np.rad2deg(self.gamma))
        v.addUnitCell({'box': {'color': 'purple'}, 'alabel': 'X', 'blabel': 'Y', 'clabel': 'Z',
                       'alabelstyle': {'fontColor': 'black', 'backgroundColor': 'white', 'inFront': True,
                                       'fontSize': 40},
                       'astyle': {'color': 'darkred', 'radius': 5, 'midpos': -10}})

        atoms = self._display_list(xmin, xmax, ymin, ymax, zmin, zmax, full_molecule=False,
                                   only_independent_atoms=False)
        m.addAtoms(atoms)
        m.setStyle({'stick': {'radius': 0.2, 'opacity': 1},
                    'sphere': {'scale': 0.3, 'colorscheme': 'jmol', 'opacity': 1}})

        if zoom:
            v.zoomTo()
        if show:
            v.show()
        else:
            v.update()

    def _widget_on_change_parameter(self, v):
        if v is not None:
            if v['name'] != 'value':
                return
        self._widget_update(zoom=True)


def create_crystal_from_cif(file, oneScatteringPowerPerElement=False,
                            connectAtoms=False, multiple=False):
    """
    Create a crystal object from a CIF file or URL
    Example:
        create_crystal_from_cif('http://www.crystallography.net/cod/2201530.cif')

    :param file: the filename/URL or python file object (need to open with mode='rb')
                 If the string begins by 'http' it is assumed that it is an URL instead,
                 e.g. from the crystallography open database
    :param oneScatteringPowerPerElement: if False (the default), then there will
          be as many ScatteringPowerAtom created as there are different elements.
          If True, only one will be created per element.
    :param connectAtoms: if True, call Crystal::ConnectAtoms to try to create
          as many Molecules as possible from the list of imported atoms.
    :param multiple: if True, all structures from the CIF will be imported, but
        the returned Crystal object and those created in the globa registry
        will not have been created in python, and so will miss the derived
        functions for display & widget.
    :return: the imported Crystal structure
    :raises: ObjCrystException - if no Crystal object can be imported
    """
    if multiple:
        if isinstance(file, str):
            if len(file) > 4:
                if file[:4].lower() == 'http':
                    return CreateCrystalFromCIF_orig(urlopen(file),
                                                     oneScatteringPowerPerElement, connectAtoms)
            with open(file, 'rb') as cif:  # Make sure file object is closed afterwards
                c = CreateCrystalFromCIF_orig(cif, oneScatteringPowerPerElement, connectAtoms)
        else:
            c = CreateCrystalFromCIF_orig(file, oneScatteringPowerPerElement, connectAtoms)
    else:
        c = Crystal()
        if isinstance(file, str):
            if len(file) > 4:
                if file[:4].lower() == 'http':
                    c.ImportCrystalFromCIF(urlopen(file),
                                           oneScatteringPowerPerElement, connectAtoms)
                    return c
            with open(file, 'rb') as cif:  # Make sure file object is closed afterwards
                c.ImportCrystalFromCIF(cif, oneScatteringPowerPerElement, connectAtoms)
        else:
            c.ImportCrystalFromCIF(file, oneScatteringPowerPerElement, connectAtoms)
    return c


# PEP8, functions should be lowercase
CreateCrystalFromCIF = create_crystal_from_cif
