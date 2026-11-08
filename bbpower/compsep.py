import math
import re
import numpy as np
import os
from scipy.linalg import sqrtm

from bbpipe import PipelineStage
from .types import FitsFile, YamlFile, DirFile
from .fg_model import FGModel
from .param_manager import ParameterManager
from .bandpasses import (Bandpass, rotate_cells, rotate_cells_mat,
                         decorrelated_bpass)
import sacc


# Log-probability for the emcee pool: a module-level function calling the
# lnprob of the instance set just before the (forked) pool is created, so
# that the workers inherit it instead of receiving the whole BBCompSep
# object pickled at every step
_EMCEE_LNPROB = None


def _emcee_lnprob(p):
    return _EMCEE_LNPROB(p)


# Same for the nautilus pool (likelihood without priors)
_NAUTILUS_LNLIKE = None


def _nautilus_lnlike(p):
    return _NAUTILUS_LNLIKE(p)


_NAUTILUS_ALLOWED = None


def _nautilus_allowed(points):
    return sum(_NAUTILUS_ALLOWED(p) for p in points)


def _fork_pool():
    """
    Pool of BBPOWER_NPROC processes (default: all cores), forked so that the
    workers inherit the instance; None without fork or with BBPOWER_NPROC=1.
    """
    import multiprocessing
    nproc = int(os.environ.get('BBPOWER_NPROC', 0)) or None
    if nproc != 1 and 'fork' in multiprocessing.get_all_start_methods():
        return multiprocessing.get_context('fork').Pool(nproc)
    return None


# free_moment_terms parsed once per set of moment roles (the regexes cost
# more than the rest of the moment terms)
_FREE_MOMENT_TERMS = {}

# "not given" for the optional moment covariance of moments_positive and
# moments_hierarchical (None is a valid covariance: no free moments)
_UNSET = object()


class BBCompSep(PipelineStage):
    """
    Component separation stage
    This stage does harmonic domain foreground cleaning (e.g. BICEP).
    The foreground model parameters are defined in the config.yml file.
    """
    name = "BBCompSep"
    inputs = [('cells_coadded', FitsFile),
              ('cells_noise', FitsFile),
              ('cells_fiducial', FitsFile),
              ('cells_coadded_cov', FitsFile)]
    outputs = [('output_dir', DirFile),
               ('config_copy', YamlFile)]
    config_options = {'likelihood_type': 'h&l', 'n_iters': 32,
                      'nwalkers': 16, 'r_init': 1.e-3,
                      'sampler': 'emcee', 'bands': 'all'}

    def setup_compsep(self):
        """
        Pre-load the data, CMB BB power spectrum, and foreground models.
        """
        self.parse_sacc_file()
        self.load_cmb()
        self.fg_model = FGModel(self.config)
        self.params = ParameterManager(self.config)
        # The Wigner-3j cube is only needed by the beta-map moments
        # (gamma_beta, amp_beta), not by the free moment spectra
        if self.config['fg_model'].get('use_moments') and any(
                'gamma_beta' in self.fg_model.components[c]['names_moments_dict']
                for c in self.fg_model.component_names):
            self.precompute_w3j()
        if self.use_handl:
            self.prepare_h_and_l()
        self.setup_fast_model()
        return

    def setup_fast_model(self):
        """
        Pre-computations of model_fast, used by model() when nothing rotates
        the polarization (no polarization angles, frequency-dependent phases,
        birefringence) and no component is decorrelated. All the bandpasses
        are joined in one frequency grid, so that each SED is evaluated once
        per call and integrated over all the bands at once; config option
        fast_model: False forces the general model.
        """
        cfg = self.config
        self.fast = (cfg.get('fast_model') is not False
                     and not cfg['cmb_model'].get('use_birefringence')
                     and not any(b.is_complex or b.do_angle or b.do_dphi1
                                 for b in self.bpss)
                     and not any(self.fg_model.components[c]['decorr']
                                 for c in self.fg_model.component_names))
        if not self.fast:
            return
        bp = self.bpss
        self._bp_len = np.array([len(b.nu) for b in bp])
        self._bp_start = np.concatenate([[0], np.cumsum(self._bp_len)[:-1]])
        self._bp_nu = np.concatenate([b.nu for b in bp])
        # bandpass weights over the CMB normalization (nu^2 apart: it
        # changes with the frequency shifts)
        self._bp_bnorm = np.concatenate([b.bnu_dnu / b.cmb_norm for b in bp])
        self._bp_shift = any(b.do_shift for b in bp)
        self._bp_gain = any(b.do_gain for b in bp)
        if not (self._bp_shift or self._bp_gain):
            self._bp_w0 = self._bp_bnorm * self._bp_nu**2
            self._cmb_scaling0 = self._bp_integrate(
                bp[0].sed_CMB_RJ(self._bp_nu) * self._bp_w0, None)
        # beta-map moments (Wigner-3j terms) in some component
        self._beta_map = [i_c for i_c, c in
                          enumerate(self.fg_model.component_names)
                          if 'gamma_beta' in
                          self.fg_model.components[c]['names_moments_dict']]
        # data vector <-> (f1, p1, f2, p2) of the model spectra
        m1, m2 = self.index_ut
        self._ut_f1, self._ut_p1 = m1 // self.npol, m1 % self.npol
        self._ut_f2, self._ut_p2 = m2 // self.npol, m2 % self.npol
        assert np.all(self.vector_indices[m1, m2] ==
                      np.arange(self.ncross)), "unexpected spectra ordering"

    def _bp_grid(self, params):
        """Frequencies and integration weights of the joined bandpasses."""
        if not self._bp_shift:
            if not self._bp_gain:
                return self._bp_nu, self._bp_w0
            nu = self._bp_nu
        else:
            dnu = np.array([params[b.name_shift] * b.nu_mean if b.do_shift
                            else 0. for b in self.bpss])
            nu = self._bp_nu + np.repeat(dnu, self._bp_len)
        return nu, self._bp_bnorm * nu**2

    def _bp_integrate(self, f, gains):
        """Integral of f (already weighted, [..., nnu]) over each band."""
        out = np.add.reduceat(f, self._bp_start, axis=-1)
        return out if gains is None else out * gains

    def _bp_gains(self, params):
        if not self._bp_gain:
            return None
        return np.array([params[b.name_gain] if b.do_gain else 1.
                         for b in self.bpss])

    def get_moments_lmax(self):
        return self.config['fg_model'].get('moments_lmax', 384)

    def precompute_w3j(self):
        from pyshtools.utils import Wigner3j  # noqa

        lmax = self.get_moments_lmax()
        ells_w3j = np.arange(0, lmax)
        w3j = np.zeros_like(ells_w3j, dtype=float)
        self.big_w3j = np.zeros((lmax, lmax, lmax))
        for ell1 in ells_w3j[1:]:
            for ell2 in ells_w3j[1:]:
                w3j_array, ellmin, ellmax = Wigner3j(ell1, ell2, 0, 0, 0)
                w3j_array = w3j_array[:ellmax - ellmin + 1]
                # make the w3j_array the same shape as the w3j
                if len(w3j_array) < len(ells_w3j):
                    reference = np.zeros(len(w3j))
                    reference[:w3j_array.shape[0]] = w3j_array
                    w3j_array = reference

                w3j_array = np.concatenate([w3j_array[-ellmin:],
                                            w3j_array[:-ellmin]])
                w3j_array = w3j_array[:len(ells_w3j)]
                w3j_array[:ellmin] = 0

                self.big_w3j[:, ell1, ell2] = w3j_array

        self.big_w3j = self.big_w3j**2

    def matrix_to_vector(self, mat):
        return mat[..., self.index_ut[0], self.index_ut[1]]

    def vector_to_matrix(self, vec):
        if vec.ndim == 1:
            mat = np.zeros([self.nmaps, self.nmaps])
            mat[self.index_ut] = vec
            mat = mat + mat.T - np.diag(mat.diagonal())
        elif vec.ndim == 2:
            mat = np.zeros([len(vec), self.nmaps, self.nmaps])
            mat[..., self.index_ut[0], self.index_ut[1]] = vec[..., :]
            for i, m in enumerate(mat):
                mat[i] = m + m.T - np.diag(m.diagonal())
        else:
            raise ValueError("Input vector can only be 1- or 2-D")
        return mat

    def _freq_pol_iterator(self):
        icl = -1
        for b1 in range(self.nfreqs):
            for p1 in range(self.npol):
                m1 = p1 + self.npol * b1
                for b2 in range(b1, self.nfreqs):
                    if b1 == b2:
                        p2_r = range(p1, self.npol)
                    else:
                        p2_r = range(self.npol)
                    for p2 in p2_r:
                        m2 = p2 + self.npol * b2
                        icl += 1
                        yield b1, b2, p1, p2, m1, m2, icl

    def parse_sacc_file(self):
        """
        Reads the data in the sacc file included the power spectra,
        bandpasses, and window functions.
        """
        # Decide if you're using H&L
        self.use_handl = self.config['likelihood_type'] == 'h&l'

        # Read data
        self.s = sacc.Sacc.load_fits(self.get_input('cells_coadded'))
        self.s_cov = sacc.Sacc.load_fits(self.get_input('cells_coadded_cov'))
        tr_comb = self.s.get_tracer_combinations()
        for tr1, tr2 in tr_comb:
            ind1 = self.s.indices(data_type='cl_bb', tracers=(tr1, tr2))
            ind2 = self.s_cov.indices(data_type='cl_bb', tracers=(tr1, tr2))
            assert np.all(ind1 == ind2), "Covariance sacc ordering is wrong"
        if self.use_handl:
            s_fid = sacc.Sacc.load_fits(self.get_input('cells_fiducial'))
            s_noi = sacc.Sacc.load_fits(self.get_input('cells_noise'))

        # Keep only desired correlations
        self.pols = self.config['pol_channels']
        corr_all = ['cl_ee', 'cl_eb', 'cl_be', 'cl_bb']
        corr_keep = []
        for m1 in self.pols:
            for m2 in self.pols:
                clname = 'cl_' + m1.lower() + m2.lower()
                corr_keep.append(clname)
        for c in corr_all:
            if c not in corr_keep:
                self.s.remove_selection(c)
                self.s_cov.remove_selection(c)
                if self.use_handl:
                    s_fid.remove_selection(c)
                    s_noi.remove_selection(c)

        # Scale cuts
        self.s.remove_selection(ell__gt=self.config['l_max'])
        self.s.remove_selection(ell__lt=self.config['l_min'])
        self.s_cov.remove_selection(ell__gt=self.config['l_max'])
        self.s_cov.remove_selection(ell__lt=self.config['l_min'])
        if self.use_handl:
            s_fid.remove_selection(ell__gt=self.config['l_max'])
            s_fid.remove_selection(ell__lt=self.config['l_min'])
            s_noi.remove_selection(ell__gt=self.config['l_max'])
            s_noi.remove_selection(ell__lt=self.config['l_min'])

        if self.config['bands'] == 'all':
            tr_names = sorted(list(self.s.tracers.keys()))
        else:
            tr_names = self.config['bands']
        self.nfreqs = len(tr_names)
        self.npol = len(self.pols)
        self.nmaps = self.nfreqs * self.npol
        self.index_ut = np.triu_indices(self.nmaps)
        self.ncross = (self.nmaps * (self.nmaps + 1)) // 2
        self.pol_order = dict(zip(self.pols, range(self.npol)))

        # Collect bandpasses
        self.bpss = []
        for i_t, tn in enumerate(tr_names):
            t = self.s.tracers[tn]
            nu = t.nu
            dnu = np.zeros_like(nu)
            dnu[1:-1] = 0.5 * (nu[2:] - nu[:-2])
            dnu[0] = nu[1] - nu[0]
            dnu[-1] = nu[-1] - nu[-2]
            bnu = t.bandpass
            self.bpss.append(Bandpass(nu, dnu, bnu, i_t+1, self.config))

        # Get ell sampling
        # Example power spectrum
        self.ell_b, _ = self.s.get_ell_cl('cl_' + 2 * self.pols[0].lower(),
                                          tr_names[0], tr_names[0])
        # Avoid l<2
        win0 = self.s.data[0]['window']
        mask_w = win0.values > 1
        self.bpw_l = win0.values[mask_w]
        self.n_ell = len(self.bpw_l)
        self.n_bpws = len(self.ell_b)
        # D_ell factor
        self.dl2cl = 2 * np.pi / (self.bpw_l * (self.bpw_l + 1))
        self.windows = np.zeros([self.ncross, self.n_bpws, self.n_ell])

        # Get power spectra and covariances
        if self.config['bands'] == 'all':
            if not (self.s_cov.covariance.covmat.shape[-1] == len(self.s.mean) == self.n_bpws * self.ncross):  # noqa: E501
                raise ValueError("C_ell vector's size is wrong")

        v2d = np.zeros([self.n_bpws, self.ncross])
        if self.use_handl:
            v2d_noi = np.zeros([self.n_bpws, self.ncross])
            v2d_fid = np.zeros([self.n_bpws, self.ncross])
        cv2d = np.zeros([self.n_bpws, self.ncross, self.n_bpws, self.ncross])

        self.vector_indices = self.vector_to_matrix(np.arange(self.ncross, dtype=int)).astype(int)  # noqa: E501
        self.indx = []

        # Parse into the right ordering
        itr1 = self._freq_pol_iterator()
        for b1, b2, p1, p2, m1, m2, ind_vec in itr1:
            t1 = tr_names[b1]
            t2 = tr_names[b2]
            pol1 = self.pols[p1].lower()
            pol2 = self.pols[p2].lower()
            cl_typ = f'cl_{pol1}{pol2}'
            ind_a = self.s.indices(cl_typ, (t1, t2))
            if len(ind_a) != self.n_bpws:
                raise ValueError("All power spectra need to be "
                                 "sampled at the same ells")
            w = self.s.get_bandpower_windows(ind_a)
            self.windows[ind_vec, :, :] = w.weight[mask_w, :].T
            v2d[:, ind_vec] = np.array(self.s.mean[ind_a])
            if self.use_handl:
                _, v2d_noi[:, ind_vec] = s_noi.get_ell_cl(cl_typ, t1, t2)
                _, v2d_fid[:, ind_vec] = s_fid.get_ell_cl(cl_typ, t1, t2)
            itr2 = self._freq_pol_iterator()
            for b1b, b2b, p1b, p2b, m1b, m2b, ind_vecb in itr2:
                t1b = tr_names[b1b]
                t2b = tr_names[b2b]
                pol1b = self.pols[p1b].lower()
                pol2b = self.pols[p2b].lower()
                cl_typb = f'cl_{pol1b}{pol2b}'
                ind_b = self.s.indices(cl_typb, (t1b, t2b))
                cv2d[:, ind_vec, :, ind_vecb] = self.s_cov.covariance.covmat[ind_a][:, ind_b]  # noqa: E501

        # Store data
        self.bbdata = self.vector_to_matrix(v2d)
        if self.use_handl:
            self.bbnoise = self.vector_to_matrix(v2d_noi)
            self.bbfiducial = self.vector_to_matrix(v2d_fid)
        self.bbcovar = cv2d.reshape([self.n_bpws * self.ncross,
                                     self.n_bpws * self.ncross])
        self.invcov = np.linalg.solve(self.bbcovar,
                                      np.identity(len(self.bbcovar)))
        return

    def load_cmb(self):
        """
        Loads the CMB BB spectrum as defined in the config file.
        """
        cmb_lensingfile = np.loadtxt(
            self.config['cmb_model']['cmb_templates'][0])
        cmb_bbfile = np.loadtxt(self.config['cmb_model']['cmb_templates'][1])

        self.cmb_ells = cmb_bbfile[:, 0]
        mask = (self.cmb_ells <= self.bpw_l.max()) & (self.cmb_ells > 1)
        self.cmb_ells = self.cmb_ells[mask]

        # TODO: this is a patch
        nell = len(self.cmb_ells)
        self.cmb_tens = np.zeros([self.npol, self.npol, nell])
        self.cmb_lens = np.zeros([self.npol, self.npol, nell])
        self.cmb_scal = np.zeros([self.npol, self.npol, nell])
        if 'B' in self.config['pol_channels']:
            ind = self.pol_order['B']
            self.cmb_tens[ind, ind] = (cmb_bbfile[:, 3][mask] -
                                       cmb_lensingfile[:, 3][mask])
            self.cmb_lens[ind, ind] = cmb_lensingfile[:, 3][mask]
        if 'E' in self.config['pol_channels']:
            ind = self.pol_order['E']
            self.cmb_tens[ind, ind] = (cmb_bbfile[:, 2][mask] -
                                       cmb_lensingfile[:, 2][mask])
            self.cmb_scal[ind, ind] = cmb_lensingfile[:, 2][mask]
        return

    def integrate_seds(self, params):
        single_sed = np.zeros([self.fg_model.n_components,
                               self.nfreqs])
        comp_scaling = np.zeros([self.fg_model.n_components,
                                 self.nfreqs, self.nfreqs])
        fg_scaling = np.zeros([self.fg_model.n_components,
                               self.fg_model.n_components,
                               self.nfreqs, self.nfreqs])
        rot_matrices = []

        for i_c, c_name in enumerate(self.fg_model.component_names):
            comp = self.fg_model.components[c_name]
            units = comp['cmb_n0_norm']
            sed_params = [params[comp['names_sed_dict'][k]]
                          for k in comp['sed'].params]
            rot_matrices.append([])

            def sed(nu):
                return comp['sed'].eval(nu, *sed_params)

            for tn in range(self.nfreqs):
                sed_b, rot = self.bpss[tn].convolve_sed(sed, params)
                single_sed[i_c, tn] = sed_b * units
                rot_matrices[i_c].append(rot)

            if comp['decorr']:
                d_amp = params[comp['decorr_param_names']['decorr_amp']]
                d_nu01 = params[comp['decorr_param_names']['decorr_nu01']]
                d_nu02 = params[comp['decorr_param_names']['decorr_nu02']]
                decorr_delta = d_amp**(1./np.log(d_nu01/d_nu02)**2)
                for f1 in range(self.nfreqs):
                    for f2 in range(f1, self.nfreqs):
                        sed_12 = decorrelated_bpass(self.bpss[f1],
                                                    self.bpss[f2],
                                                    sed, params,
                                                    decorr_delta)
                        comp_scaling[i_c, f1, f2] = sed_12 * units * units
            else:
                comp_scaling[i_c] = np.outer(single_sed[i_c], single_sed[i_c])

        for i_c1, c_name1 in enumerate(self.fg_model.component_names):
            fg_scaling[i_c1, i_c1] = comp_scaling[i_c1]
            for c_name2, epsname in self.fg_model.components[c_name1]['names_x_dict'].items():  # noqa: E501
                i_c2 = self.fg_model.component_order[c_name2]
                eps = params[epsname]
                fg_scaling[i_c1, i_c2] = eps * np.outer(single_sed[i_c1],
                                                        single_sed[i_c2])
                fg_scaling[i_c2, i_c1] = eps * np.outer(single_sed[i_c2],
                                                        single_sed[i_c1])
        return fg_scaling, np.array(rot_matrices)

    def evaluate_power_spectra(self, params):
        fg_pspectra = np.zeros([self.fg_model.n_components, self.npol,
                                self.npol, self.n_ell])

        # Fill diagonal
        for i_c, c_name in enumerate(self.fg_model.component_names):
            comp = self.fg_model.components[c_name]
            for cl_comb, clfunc in comp['cl'].items():
                m1, m2 = cl_comb
                ip1 = self.pol_order[m1]
                ip2 = self.pol_order[m2]
                pspec_params = [params[comp['names_cl_dict'][cl_comb][k]]
                                for k in clfunc.params]
                p_spec = clfunc.eval(self.bpw_l, *pspec_params) * self.dl2cl
                fg_pspectra[i_c, ip1, ip2] = p_spec
                if m1 != m2:
                    fg_pspectra[i_c, ip2, ip1] = p_spec

        return fg_pspectra

    def model(self, params):
        """
        Defines the total model and integrates over
        the bandpasses and windows: model_fast when possible (see
        setup_fast_model), model_general otherwise.
        """
        if getattr(self, 'fast', False):
            return self.model_fast(params)
        return self.model_general(params)

    def model_fast(self, params):
        """
        Same model as model_general without polarization rotations and
        decorrelation, vectorized over frequencies and components.
        """
        nf, nc, npol = self.nfreqs, self.fg_model.n_components, self.npol
        comps = [self.fg_model.components[c]
                 for c in self.fg_model.component_names]
        # [nell, npol, npol]
        cmb_cell = np.transpose((params['r_tensor'] * self.cmb_tens +
                                 params['A_lens'] * self.cmb_lens +
                                 self.cmb_scal) * self.dl2cl, axes=[2, 0, 1])

        # SEDs on the joined bandpasses, weighted for the integration
        nu, w = self._bp_grid(params)
        gains = self._bp_gains(params)
        if self._bp_shift or self._bp_gain:
            cmb_scaling = self._bp_integrate(
                self.bpss[0].sed_CMB_RJ(nu) * w, gains)
        else:
            cmb_scaling = self._cmb_scaling0
        sed_w = []
        single_sed = np.zeros([nc, nf])
        for i_c, comp in enumerate(comps):
            sed_params = [params[comp['names_sed_dict'][k]]
                          for k in comp['sed'].params]
            sed_w.append(comp['sed'].eval(nu, *sed_params) * w)
            single_sed[i_c] = (self._bp_integrate(sed_w[i_c], gains) *
                               comp['cmb_n0_norm'])
        # [ncomp, ncomp, nfreq, nfreq]
        fg_scaling = np.zeros([nc, nc, nf, nf])
        for i_c1, comp in enumerate(comps):
            fg_scaling[i_c1, i_c1] = np.outer(single_sed[i_c1],
                                              single_sed[i_c1])
            for c_name2, epsname in comp['names_x_dict'].items():
                i_c2 = self.fg_model.component_order[c_name2]
                eps = params[epsname]
                fg_scaling[i_c1, i_c2] = eps * np.outer(single_sed[i_c1],
                                                        single_sed[i_c2])
                fg_scaling[i_c2, i_c1] = eps * np.outer(single_sed[i_c2],
                                                        single_sed[i_c1])

        # [ncomp, nell, npol, npol]
        fg_cell = np.transpose(self.evaluate_power_spectra(params),
                               axes=[0, 3, 1, 2])
        # spectra of the component pairs: auto, and for the cross the
        # geometric mean of the diagonals (EB term zero)
        # [ncomp, ncomp, nell, npol, npol]
        pair_cell = np.zeros((nc, nc) + fg_cell.shape[1:])
        diag = np.einsum('clii->cli', fg_cell)
        for c1 in range(nc):
            for c2 in range(nc):
                if c1 == c2:
                    pair_cell[c1, c1] = fg_cell[c1]
                else:
                    pair_cell[c1, c2][:, np.arange(npol), np.arange(npol)] = \
                        np.sqrt(diag[c1] * diag[c2])
        # [nfreq, nfreq, nell, npol, npol]
        cls_array_fg = (np.outer(cmb_scaling, cmb_scaling).reshape(
            nf, nf, 1, 1, 1) * cmb_cell +
            (fg_scaling.reshape(nc * nc, nf * nf).T @
             pair_cell.reshape(nc * nc, -1)).reshape((nf, nf) +
                                                     fg_cell.shape[1:]))

        if self.config['fg_model'].get('use_moments'):
            if self._beta_map:
                self._add_beta_map_moments(params, comps, nu, sed_w, gains,
                                           fg_scaling, fg_cell, cls_array_fg)
            self._add_free_moments_fast(params, nu, w, sed_w, gains,
                                        cls_array_fg)

        # Window convolution, all the spectra (upper triangle) at once
        # [ncross, nell]
        cls_vec = cls_array_fg[self._ut_f1, self._ut_f2, :,
                               self._ut_p1, self._ut_p2]
        # [ncross, nbpw]
        clband = (self.windows @ cls_vec[:, :, None])[..., 0]
        out = np.zeros([self.n_bpws, self.nmaps, self.nmaps])
        out[:, self.index_ut[0], self.index_ut[1]] = clband.T
        out[:, self.index_ut[1], self.index_ut[0]] = clband.T
        return out

    def _add_beta_map_moments(self, params, comps, nu, sed_w, gains,
                              fg_scaling, fg_cell, cls_array_fg):
        """1x1 and 0x2 terms of the beta-map moments (as model_general)."""
        nf, nc = self.nfreqs, self.fg_model.n_components
        # [nfreq, ncomp]
        d1 = np.zeros([nf, nc])
        d2 = np.zeros([nf, nc])
        for i_c, comp in enumerate(comps):
            x = np.log(nu / params[comp['names_sed_dict']['nu0']])
            d1[:, i_c] = (self._bp_integrate(x * sed_w[i_c], gains) *
                          comp['cmb_n0_norm'])
            d2[:, i_c] = (self._bp_integrate(x**2 * sed_w[i_c], gains) *
                          comp['cmb_n0_norm'])
        lmax_mom = self.get_moments_lmax()
        cls_11 = np.zeros_like(fg_cell)
        cls_02 = np.zeros_like(fg_cell)
        for i_c in self._beta_map:
            names = comps[i_c]['names_moments_dict']
            cl_betas = self.bcls(lmax=lmax_mom,
                                 gamma=params[names['gamma_beta']],
                                 amp=params[names['amp_beta']] * 1E-6)
            cls_11[i_c, :lmax_mom] = self.evaluate_1x1(
                params, lmax=lmax_mom, cls_cc=fg_cell[i_c], cls_bb=cl_betas)
            cls_02[i_c, :lmax_mom] = self.evaluate_0x2(
                params, lmax=lmax_mom, cls_cc=fg_cell[i_c], cls_bb=cl_betas)
        # [nfreq, ncomp]
        sq = np.sqrt(np.einsum('ccff->fc', fg_scaling))
        cls_array_fg += np.einsum('ic,jc,clpq->ijlpq', d1, d1, cls_11)
        cls_array_fg += 0.5 * (np.einsum('ic,jc,clpq->ijlpq', d2, sq, cls_02) +
                               np.einsum('jc,ic,clpq->ijlpq', d2, sq, cls_02))

    def _add_free_moments_fast(self, params, nu, w, sed_w, gains,
                               cls_array_fg):
        """add_free_moments with the SEDs already on the joined bandpasses."""
        if 'B' not in self.pol_order:
            return
        ipb = self.pol_order['B']
        lx = np.log(self.bpw_l / 80.)
        cache = {}

        def c_nm(c_name, k):
            if (c_name, k) not in cache:
                cache[(c_name, k)] = self._c_nm_fast(params, c_name, k, nu,
                                                     w, sed_w, gains)
            return cache[(c_name, k)]

        for c_name in self.fg_model.component_names:
            names_mom = self.fg_model.components[c_name]['names_moments_dict']
            for tag, a, b, other in self.free_moment_terms(names_mom):
                amp = params[names_mom['A_' + tag]]
                if amp == 0:
                    continue
                slope = (params[names_mom['alpha_' + tag]]
                         if 'alpha_' + tag in names_mom else 0.)
                cl = amp * np.exp(slope * lx) * self.dl2cl
                wm = np.outer(c_nm(c_name, a), c_nm(other or c_name, b))
                if a != b or other is not None:
                    wm = wm + wm.T
                cls_array_fg[:, :, :, ipb, ipb] += wm[:, :, None] * cl

    def _c_nm_fast(self, params, c_name, k, nu, w, sed_w, gains, eps=1e-3):
        """integrate_seds_der_bp on the joined bandpasses."""
        n, m = k
        i_c = self.fg_model.component_order[c_name]
        comp = self.fg_model.components[c_name]
        units = comp['cmb_n0_norm']
        x = np.log(nu / params[comp['names_sed_dict']['nu0']])
        norm = 1. / (math.factorial(n) * math.factorial(m))
        if m == 0:
            f = x**n * sed_w[i_c]
        else:
            if 'temp' not in comp['names_sed_dict']:
                raise ValueError(f"{c_name}: temperature moments need a "
                                 "'temp' SED parameter")
            sed = self._sed_T(c_name)
            keys = sed.params
            sed_params = [params[comp['names_sed_dict'][kk]] for kk in keys]
            i_t = keys.index('temp')
            p0 = 1. / sed_params[i_t]
            h = eps * p0

            def sed_at(p):
                q = list(sed_params)
                q[i_t] = 1. / p
                return sed.eval(nu, *q)

            fp, fm = sed_at(p0 + h), sed_at(p0 - h)
            if m == 1:
                d = (fp - fm) / (2 * h)
            else:
                d = (fp - 2 * sed_at(p0) + fm) / h**2
            f = x**n * d * w
        return self._bp_integrate(f, gains) * units * norm

    def model_general(self, params):
        """
        Defines the total model and integrates over
        the bandpasses and windows (any configuration).
        """
        # [npol,npol,nell]
        cmb_cell = (params['r_tensor'] * self.cmb_tens +
                    params['A_lens'] * self.cmb_lens +
                    self.cmb_scal) * self.dl2cl
        # [nell,npol,npol]
        cmb_cell = np.transpose(cmb_cell, axes=[2, 0, 1])
        if self.config['cmb_model'].get('use_birefringence'):
            bi_angle = np.radians(params['birefringence'])
            c = np.cos(2*bi_angle)
            s = np.sin(2*bi_angle)
            bmat = np.array([[c, s],
                             [-s, c]])
            cmb_cell = rotate_cells_mat(bmat, bmat, cmb_cell)
        
        # [ncomp, ncomp, nfreq, nfreq], [ncomp, nfreq,[matrix]]
        fg_scaling, rot_m = self.integrate_seds(params)
        # [ncomp,npol,npol,nell]
        fg_cell = self.evaluate_power_spectra(params)

        # Add all components scaled in frequency (and HWP-rotated if needed)
        # [nfreq, nfreq, nell, npol, npol]
        cls_array_fg = np.zeros([self.nfreqs, self.nfreqs,
                                 self.n_ell, self.npol, self.npol])
        # [ncomp,nell,npol,npol]
        fg_cell = np.transpose(fg_cell, axes=[0, 3, 1, 2])

        # SED scaling
        cmb_scaling = np.ones(self.nfreqs)
        cmb_rot = []
        for f1 in range(self.nfreqs):
            cs, crot = self.bpss[f1].convolve_sed(None, params)
            cmb_scaling[f1] = cs
            cmb_rot.append(crot)

        for f1 in range(self.nfreqs):
            # Note that we only need to fill in half of the frequencies
            for f2 in range(f1, self.nfreqs):
                cls = (rotate_cells_mat(cmb_rot[f2], cmb_rot[f1], cmb_cell) *
                       cmb_scaling[f1] * cmb_scaling[f2])

                # Loop over component pairs
                for c1 in range(self.fg_model.n_components):
                    for c2 in range(self.fg_model.n_components):
                        mat1 = rot_m[c1, f1]
                        mat2 = rot_m[c2, f2]
                        if c1 == c2:
                            clrot = rotate_cells_mat(mat2, mat1, fg_cell[c1])
                        else:
                            # For cross component, enforcing EB term is zero.
                            cl_cross = np.zeros((self.n_ell,
                                                 self.npol, self.npol))
                            for i in range(self.npol):
                                cl_cross[:, i, i] = np.sqrt(
                                    fg_cell[c1, :, i, i] * fg_cell[c2, :, i, i]
                                )
                            clrot = rotate_cells_mat(mat2, mat1, cl_cross)
                        cls += clrot * fg_scaling[c1, c2, f1, f2]
                cls_array_fg[f1, f2] = cls

        # Add moment terms if needed
        if self.config['fg_model'].get('use_moments'):
            # TODO: moments work with:
            # - B-only
            # - No polarization angle business
            # - Only power-law beta power spectra at l_pivot=80

            # Evaluate 1st/2nd order SED derivatives.
            # [nfreq, ncomp]
            fg_scaling_d1 = self.integrate_seds_der(params, order=1)
            fg_scaling_d2 = self.integrate_seds_der(params, order=2)

            # Compute 1x1 for each component
            # Compute 0x2 for each component (essentially this is sigma_beta)
            # Evaluate beta power spectra.
            lmax_mom = self.get_moments_lmax()
            # [ncomp, nell, npol, npol]
            cls_11 = np.zeros([self.fg_model.n_components, self.n_ell,
                               self.npol, self.npol])
            # [ncomp, nell, npol, npol]
            cls_02 = np.zeros([self.fg_model.n_components, self.n_ell,
                               self.npol, self.npol])
            for i_c, c_name in enumerate(self.fg_model.component_names):
                comp = self.fg_model.components[c_name]
                if 'gamma_beta' not in comp['names_moments_dict']:
                    continue
                gamma = params[comp['names_moments_dict']['gamma_beta']]
                amp = params[comp['names_moments_dict']['amp_beta']] * 1E-6
                cl_betas = self.bcls(lmax=lmax_mom, gamma=gamma, amp=amp)
                cl_cc = fg_cell[i_c, :]
                # cls_1x1 = 0
                cls_1x1 = self.evaluate_1x1(params, lmax=lmax_mom,
                                            cls_cc=cl_cc,
                                            cls_bb=cl_betas)
                cls_11[i_c, :lmax_mom, :, :] = cls_1x1
                # cls_0x2 = 0
                cls_0x2 = self.evaluate_0x2(params, lmax=lmax_mom,
                                            cls_cc=cl_cc,
                                            cls_bb=cl_betas)
                cls_02[i_c, :lmax_mom, :, :] = cls_0x2

            # Add components scaled in frequency
            for f1 in range(self.nfreqs):
                # Note that we only need to fill in half of the frequencies
                for f2 in range(f1, self.nfreqs):
                    cls = np.zeros([self.n_ell, self.npol, self.npol])
                    for c1 in range(self.fg_model.n_components):
                        cls += (fg_scaling_d1[f1, c1] * fg_scaling_d1[f2, c1] *
                                cls_11[c1])
                        cls += 0.5 * (fg_scaling_d2[f1, c1] *
                                      (fg_scaling[c1, c1, f2, f2])**0.5 +
                                      fg_scaling_d2[f2, c1] *
                                      (fg_scaling[c1, c1, f1, f1])**0.5) * cls_02[c1]  # noqa: E501
                    cls_array_fg[f1, f2] += cls

            # Free moment spectra (BB only)
            self.add_free_moments(params, cls_array_fg)

        # Window convolution
        cls_array_list = np.zeros([self.n_bpws, self.nfreqs,
                                   self.npol, self.nfreqs,
                                   self.npol])
        for f1 in range(self.nfreqs):
            for p1 in range(self.npol):
                m1 = f1*self.npol+p1
                for f2 in range(f1, self.nfreqs):
                    p0 = p1 if f1 == f2 else 0
                    for p2 in range(p0, self.npol):
                        m2 = f2*self.npol+p2
                        windows = self.windows[self.vector_indices[m1, m2]]
                        clband = np.dot(windows, cls_array_fg[f1, f2, :,
                                                              p1, p2])
                        cls_array_list[:, f1, p1, f2, p2] = clband
                        if m1 != m2:
                            cls_array_list[:, f2, p2, f1, p1] = clband

        # Polarization angle rotation
        for f1 in range(self.nfreqs):
            for f2 in range(self.nfreqs):
                cls_array_list[:, f1, :, f2, :] = rotate_cells(self.bpss[f2],
                                                               self.bpss[f1],
                                                               cls_array_list[:, f1, :, f2, :],  # noqa: E501
                                                               params)

        return cls_array_list.reshape([self.n_bpws, self.nmaps, self.nmaps])

    def bcls(self, lmax, gamma, amp):
        ls = np.arange(lmax)
        bcls = np.zeros(len(ls))
        bcls[2:] = (ls[2:] / 80.)**gamma
        return bcls*amp

    def integrate_seds_der(self, params, order=1):
        """
        Define the first order derivative of the SED
        """
        fg_scaling_der = np.zeros([self.fg_model.n_components,
                                   self.nfreqs])

        for i_c, c_name in enumerate(self.fg_model.component_names):
            comp = self.fg_model.components[c_name]
            units = comp['cmb_n0_norm']
            sed_params = [params[comp['names_sed_dict'][k]]
                          for k in comp['sed'].params]

            # Set SED function with scaling beta
            def sed_der(nu):
                nu0 = params[comp['names_sed_dict']['nu0']]
                x = np.log(nu / nu0)
                # This is only valid for spectral indices
                return x**order * comp['sed'].eval(nu, *sed_params)

            for tn in range(self.nfreqs):
                sed_b = self.bpss[tn].convolve_sed(sed_der, params)[0]
                fg_scaling_der[i_c, tn] = sed_b * units

        return fg_scaling_der.T

    @staticmethod
    def free_moment_terms(names_mom):
        """
        Free moment spectra of a component: (tag, (n1, m1), (n2, m2), other)
        for each A_w<n1><m1>w<n2><m2>[_x_<other>] role in its moments block,
        i.e. the cross spectrum of the moment maps w_nm = A * dbeta^n * dp^m
        (p = 1/T, n + m <= 2) of this component (other = None) or of this
        component and component `other`.
        """
        key = tuple(sorted(names_mom))
        if key not in _FREE_MOMENT_TERMS:
            _FREE_MOMENT_TERMS[key] = BBCompSep._parse_free_moment_terms(key)
        return _FREE_MOMENT_TERMS[key]

    @staticmethod
    def _parse_free_moment_terms(names_mom):
        terms = []
        for role in names_mom:
            m = re.fullmatch(r'A_w(\d)(\d)w(\d)(\d)(?:_x_(\w+))?', role)
            if m is None:
                continue
            a = (int(m.group(1)), int(m.group(2)))
            b = (int(m.group(3)), int(m.group(4)))
            other = m.group(5)
            if sum(a) > 2 or sum(b) > 2:
                raise ValueError(f"free moment {role}: maps up to order 2")
            if a == b == (0, 0):
                raise ValueError(f"free moment {role}: the 0x0 term is the "
                                 "component itself (or epsilon)")
            terms.append((role[2:], a, b, other))
        return sorted(terms, key=lambda t: t[0])

    def _sed_T(self, c_name):
        """
        Copy of the SED of a component with the temperature always free
        (fgbuster drops fixed parameters from the SED signature).
        """
        comp = self.fg_model.components[c_name]
        if 'sed_T' not in comp:
            import fgbuster.component_model as fgc
            from .fg_model import get_function
            cfg_c = self.config['fg_model'][c_name]
            kw = {}
            for k, l in cfg_c['sed_parameters'].items():
                fixed = l[1] == 'fixed' and l[0] != 'temp'
                kw[l[0]] = l[2][0] if fixed else None
            comp['sed_T'] = get_function(fgc, cfg_c['sed'])(
                **kw, units='K_RJ')
        return comp['sed_T']

    def integrate_seds_der_bp(self, params, c_name, n, m, eps=1e-3):
        """
        Bandpass-integrated d^n/dbeta^n d^m/dp^m SED / (n! m!) of a
        component, p = 1/T [1/K] (central finite differences in p, m <= 2).
        [nfreq]
        """
        comp = self.fg_model.components[c_name]
        units = comp['cmb_n0_norm']
        nu0 = params[comp['names_sed_dict']['nu0']]
        norm = 1. / (math.factorial(n) * math.factorial(m))
        if m == 0:
            sed = comp['sed']
            sed_params = [params[comp['names_sed_dict'][k]]
                          for k in sed.params]

            def sed_der(nu):
                return np.log(nu / nu0)**n * sed.eval(nu, *sed_params)
        else:
            if 'temp' not in comp['names_sed_dict']:
                raise ValueError(f"{c_name}: temperature moments need a "
                                 "'temp' SED parameter")
            sed = self._sed_T(c_name)
            keys = sed.params
            sed_params = [params[comp['names_sed_dict'][k]] for k in keys]
            i_t = keys.index('temp')
            p0 = 1. / sed_params[i_t]
            h = eps * p0

            def sed_at(nu, p):
                q = list(sed_params)
                q[i_t] = 1. / p
                return sed.eval(nu, *q)

            def sed_der(nu):
                fp, fm = sed_at(nu, p0 + h), sed_at(nu, p0 - h)
                if m == 1:
                    d = (fp - fm) / (2 * h)
                else:
                    d = (fp - 2 * sed_at(nu, p0) + fm) / h**2
                return np.log(nu / nu0)**n * d

        return np.array([self.bpss[tn].convolve_sed(sed_der, params)[0]
                         for tn in range(self.nfreqs)]) * units * norm

    def moment_covariance(self, params):
        """
        Covariance of the moment maps implied by the free moment spectra, at
        every multipole of the data (l_min <= ell <= l_max): returns
        ({(component, (n, m)): index}, matrix [nell, nmaps, nmaps]), or None
        if no component has free moment spectra. w00 x w00 is the
        component's own spectrum, the 0x0 cross between components
        epsilon * sqrt(C C'); pairs without a free spectrum are 0.
        """
        if 'B' not in self.pol_order:
            return None
        ipb = self.pol_order['B']
        sel = ((self.bpw_l >= self.config['l_min']) &
               (self.bpw_l <= self.config['l_max']))
        lx = np.log(self.bpw_l[sel] / 80.)
        dl2cl = np.broadcast_to(self.dl2cl, self.bpw_l.shape)[sel]
        fg_cell = None
        idx, entries = {}, {}

        def own(c_name):
            nonlocal fg_cell
            if fg_cell is None:
                fg_cell = self.evaluate_power_spectra(params)
            i_c = self.fg_model.component_order[c_name]
            return fg_cell[i_c, ipb, ipb][sel]

        def key(c_name, k):
            if (c_name, k) not in idx:
                idx[(c_name, k)] = len(idx)
                if k == (0, 0):
                    i = idx[(c_name, k)]
                    entries[(i, i)] = own(c_name)
                else:
                    key(c_name, (0, 0))
            return idx[(c_name, k)]

        def add(i, j, cl):
            i, j = min(i, j), max(i, j)
            entries[(i, j)] = entries.get((i, j), 0.) + cl

        for c_name in self.fg_model.component_names:
            names_mom = self.fg_model.components[c_name]['names_moments_dict']
            for tag, a, b, other in self.free_moment_terms(names_mom):
                amp = params[names_mom['A_' + tag]]
                if amp == 0:
                    continue
                slope = (params[names_mom['alpha_' + tag]]
                         if 'alpha_' + tag in names_mom else 0.)
                cl = amp * np.exp(slope * lx) * dl2cl
                add(key(c_name, a), key(other or c_name, b), cl)
        if not idx:
            return None

        # 0x0 cross between components present in the matrix
        for c1 in self.fg_model.component_names:
            for c2, eps in self.fg_model.components[c1]['names_x_dict'].items():
                if (c1, (0, 0)) in idx and (c2, (0, 0)) in idx:
                    i, j = idx[(c1, (0, 0))], idx[(c2, (0, 0))]
                    add(i, j, params[eps] * np.sqrt(entries[(i, i)] *
                                                    entries[(j, j)]))

        n = len(idx)
        mat = np.zeros([sel.sum(), n, n])
        for (i, j), cl in entries.items():
            mat[:, i, j] = cl
            mat[:, j, i] = cl
        return idx, mat

    def moments_positive(self, params, cov=_UNSET):
        """
        True if, at every multipole of the data, the covariance matrix of the
        moment maps (moment_covariance) is positive semi-definite. Only
        components with free moment spectra (and their cross partners)
        enter. Enabled by fg_model: moments_positive: True. `cov`: the
        output of moment_covariance(params), if already computed.
        """
        if cov is _UNSET:
            cov = self.moment_covariance(params)
        if cov is None:
            return True
        mat = cov[1]
        # scale to unit diagonal where possible (eigenvalues of comparable
        # size), then require non-negative eigenvalues
        d = np.sqrt(np.abs(np.einsum('lii->li', mat)))
        d[d == 0] = 1.
        mat = mat / d[:, :, None] / d[:, None, :]
        return bool(np.all(np.linalg.eigvalsh(mat)[:, 0] >= -1e-10))

    def moments_hierarchical(self, params, k, cov=_UNSET):
        """
        True if, at every multipole of the data and for every component, the
        second-order moment auto-spectra are bounded by the first-order ones:

            C[w20 x w20] <= k C[w10 x w10]^2 / C[w00 x w00]   (beta)
            C[w02 x w02] <= k C[w01 x w01]^2 / C[w00 x w00]   (p = 1/T)
            C[w11 x w11] <= k C[w10 x w10] C[w01 x w01] / C[w00 x w00]

        For a Gaussian delta-beta uncorrelated with the amplitude the ratio
        is ~3 (<dbeta^4> = 3 sigma^4). A second-order map without its
        first-order auto-spectrum must then have zero power. Enabled by
        fg_model: moments_hierarchy: k. `cov`: the output of
        moment_covariance(params), if already computed.
        """
        if cov is _UNSET:
            cov = self.moment_covariance(params)
        if cov is None:
            return True
        idx, mat = cov

        def auto(c_name, kk):
            i = idx.get((c_name, kk))
            return None if i is None else mat[:, i, i]

        for c_name in self.fg_model.component_names:
            c00 = auto(c_name, (0, 0))
            if c00 is None:
                continue
            for second, f1, f2 in [((2, 0), (1, 0), (1, 0)),
                                   ((0, 2), (0, 1), (0, 1)),
                                   ((1, 1), (1, 0), (0, 1))]:
                c2 = auto(c_name, second)
                if c2 is None:
                    continue
                a1 = auto(c_name, f1)
                a2 = auto(c_name, f2)
                a1 = 0. if a1 is None else a1
                a2 = 0. if a2 is None else a2
                if np.any(c2 > k * a1 * a2 / c00 * (1 + 1e-10)):
                    return False
        return True

    def add_free_moments(self, params, cls_array_fg):
        """
        Moment expansion with free cross spectra of the moment maps,
        following Vacher et al. 2022 (A&A 660, A111), with a power law in
        ell for each moment spectrum instead of a free value per ell bin,
        global beta0 and T0, and moments in p = 1/T.

        With w_nm = A * (beta - beta0)^n * (p - p0)^m the moment maps of a
        component (p = 1/T) and c_nm(nu) = <d^n_beta d^m_p SED> / (n! m!)
        (bandpass-integrated), the BB spectrum gets, for each pair of maps
        a != b and a == b respectively,

            C^{ab}_ell [c_a(nu1) c_b(nu2) + c_b(nu1) c_a(nu2)]
            C^{aa}_ell  c_a(nu1) c_a(nu2)

        with C^{ab} = A_w<a>w<b> (ell/80)^alpha_w<a>w<b>, in D_ell units at
        nu0 (same units as the amplitude of the component; dbeta
        dimensionless, dp in 1/K). alpha is optional (default 0). Cross
        terms (a != b) can be negative.

        Cross spectra between the moment maps of two components (role
        A_w<a>w<b>_x_<other component> in the moments block of the first:
        map a of this component times map b of the other) enter with
        C^{ab}_ell [c_a(nu1) c'_b(nu2) + c'_b(nu1) c_a(nu2)], in the units
        of sqrt(A A') (same convention as epsilon * sqrt(C C')).

        Independent of the beta-map moments
        (gamma_beta, amp_beta); BB only, no polarization rotation.
        """
        if 'B' not in self.pol_order:
            return
        ipb = self.pol_order['B']
        lx = np.log(self.bpw_l / 80.)
        cache = {}

        def c_nm(c_name, k):
            if (c_name, k) not in cache:
                cache[(c_name, k)] = self.integrate_seds_der_bp(params, c_name,
                                                               *k)
            return cache[(c_name, k)]

        for c_name in self.fg_model.component_names:
            names_mom = self.fg_model.components[c_name]['names_moments_dict']
            for tag, a, b, other in self.free_moment_terms(names_mom):
                amp = params[names_mom['A_' + tag]]
                if amp == 0:
                    continue
                slope = (params[names_mom['alpha_' + tag]]
                         if 'alpha_' + tag in names_mom else 0.)
                cl = amp * np.exp(slope * lx) * self.dl2cl
                w = np.outer(c_nm(c_name, a), c_nm(other or c_name, b))
                if a != b or other is not None:
                    w = w + w.T
                for f1 in range(self.nfreqs):
                    for f2 in range(f1, self.nfreqs):
                        cls_array_fg[f1, f2, :, ipb, ipb] += w[f1, f2] * cl

    def evaluate_1x1(self, params, lmax, cls_cc, cls_bb):
        """
        Evaluate the 1x1 moment for auto-spectra
        """

        ls = np.arange(lmax)
        v_left = (2*ls+1)[:, None, None] * cls_cc[:lmax, :, :]
        v_right = (2*ls+1) * cls_bb[:lmax]

        mat = self.big_w3j
        v_left = np.transpose(v_left, axes=[1, 0, 2])
        moment1x1 = np.dot(np.dot(mat, v_right), v_left) / (4*np.pi)
        return moment1x1

    def evaluate_0x2(self, params, lmax, cls_cc, cls_bb):
        """
        Evaluate the 0x2 moment for auto-spectra
        Assume power law for beta
        """
        ls = np.arange(lmax)
        prefac = np.sum((2 * ls + 1) * cls_bb) / (4*np.pi)
        return cls_cc[:lmax] * prefac

    def chi_sq_dx(self, params):
        """
        Chi^2 likelihood.
        """
        model_cls = self.model(params)
        return self.matrix_to_vector(self.bbdata - model_cls).flatten()

    def prepare_h_and_l(self):
        fiducial_noise = self.bbfiducial + self.bbnoise
        self.Cfl_sqrt = np.array([sqrtm(f) for f in fiducial_noise])
        self.observed_cls = self.bbdata + self.bbnoise
        return

    def h_and_l_dx(self, params):
        """
        Hamimeche and Lewis likelihood.
        Taken from Cobaya written by H, L and Torrado
        See: https://github.com/CobayaSampler/cobaya/blob/master/cobaya/likelihoods/_cmblikes_prototype/_cmblikes_prototype.py  # noqa: E501
        """
        # all bandpowers at once (same algebra as h_and_l below)
        C = self.model(params) + self.bbnoise
        try:
            d, U = np.linalg.eigh(C)
        except np.linalg.LinAlgError:
            return [np.inf]
        if np.any(d <= 0):
            return [np.inf]
        Ut = np.swapaxes(U, -1, -2)
        rot = Ut @ self.observed_cls @ U
        roots = np.sqrt(d)
        rot /= roots[:, :, None] * roots[:, None, :]
        rot = U @ rot @ Ut
        try:
            d, rot = np.linalg.eigh(rot)
        except np.linalg.LinAlgError:
            return [np.inf]
        if np.any(d <= 0):
            return [np.inf]
        g = np.sign(d - 1) * np.sqrt(2 * np.maximum(0, d - np.log(d) - 1))
        U = self.Cfl_sqrt @ rot
        X = (U * g[:, None, :]) @ np.swapaxes(U, -1, -2)
        return self.matrix_to_vector(X).flatten()

    def h_and_l(self, C, Chat, Cfl_sqrt):
        try:
            diag, U = np.linalg.eigh(C)
        except:  # noqa
            return [np.inf]
        rot = U.T.dot(Chat).dot(U)
        roots = np.sqrt(diag)
        for i, root in enumerate(roots):
            rot[i, :] /= root
            rot[:, i] /= root
        U.dot(rot.dot(U.T), rot)
        try:
            diag, rot = np.linalg.eigh(rot)
        except:  # noqa
            return [np.inf]
        diag = (np.sign(diag - 1) *
                np.sqrt(2 * np.maximum(0, diag - np.log(diag) - 1)))
        Cfl_sqrt.dot(rot, U)
        for i, d in enumerate(diag):
            rot[:, i] = U[:, i] * d
        return rot.dot(U.T)

    def lnprob(self, par):
        """
        Likelihood with priors.
        """
        prior = self.params.lnprior(par)
        if not np.isfinite(prior):
            return -np.inf

        return prior + self.lnlike(par)

    def moment_constraints(self):
        """True if moments_positive or moments_hierarchy is enabled."""
        fgm = self.config['fg_model']
        return bool(fgm.get('use_moments') and
                    (fgm.get('moments_positive') or
                     fgm.get('moments_hierarchy')))

    def moments_allowed(self, params):
        """
        True if the parameters satisfy the enabled moment constraints
        (moments_positive, moments_hierarchy). Outside them the likelihood
        is -inf, i.e. the constraints are part of the prior.
        """
        if not self.moment_constraints():
            return True
        fgm = self.config['fg_model']
        # computed once for both constraints
        cov = self.moment_covariance(params)
        if (fgm.get('moments_positive')
                and not self.moments_positive(params, cov)):
            return False
        if (fgm.get('moments_hierarchy')
                and not self.moments_hierarchical(
                    params, fgm['moments_hierarchy'], cov)):
            return False
        return True

    def lnlike(self, par):
        """
        Likelihood without priors.
        """
        params = self.params.build_params(par)
        if not self.moments_allowed(params):
            return -np.inf
        if self.use_handl:
            dx = self.h_and_l_dx(params)
            if np.any(np.isinf(dx)):
                return -np.inf
        else:
            dx = self.chi_sq_dx(params)
        like = -0.5 * np.dot(dx, np.dot(self.invcov, dx))
        
        return like

    def emcee_sampler(self):
        """
        Sample the model with MCMC.
        """
        import emcee

        fname_temp = self.get_output('output_dir')+'/emcee.npz.h5'
        backend = emcee.backends.HDFBackend(fname_temp)

        nwalkers = self.config['nwalkers']
        n_iters = self.config['n_iters']
        ndim = len(self.params.p0)
        found_file = os.path.isfile(fname_temp)

        try:
            nchain = len(backend.get_chain())
        except AttributeError:
            found_file = False

        if not found_file:
            backend.reset(nwalkers, ndim)
            pos = [self.params.p0 + 1.e-3*np.random.randn(ndim)
                   for i in range(nwalkers)]
            nsteps_use = n_iters
        else:
            print("Restarting from previous run")
            pos = None
            nsteps_use = max(n_iters-nchain, 0)

        # BBPOWER_NPROC: number of pool processes (e.g. several runs per
        # node on a cluster); default: all cores. The pool is forked after
        # the setup (the workers inherit the instance); without fork (or
        # with BBPOWER_NPROC=1) emcee runs serially
        global _EMCEE_LNPROB
        _EMCEE_LNPROB = self.lnprob
        pool = _fork_pool()
        try:
            import time
            start = time.time()
            sampler = emcee.EnsembleSampler(nwalkers, ndim,
                                            _emcee_lnprob, pool=pool,
                                            backend=backend)
            if nsteps_use > 0:
                sampler.run_mcmc(pos, nsteps_use, store=True, progress=False)
            end = time.time()
        finally:
            if pool is not None:
                pool.close()
                pool.join()

        return sampler, end-start

    def _prior_fraction_cache(self, n):
        """
        Cache of the allowed prior fraction (moment constraints) in the
        directory BBPOWER_PRIOR_CACHE (none if unset): the fraction depends
        only on the foreground priors and constraints, the multipoles and
        n, not on the data nor on the CMB parameters, so it is shared by the
        same model in every set and by its r-free / r = 0 versions. Key:
        the fg_model config without the tophat centres (starting points
        only), pol channels, l_min, l_max, the multipoles, n. One file per
        key, written atomically. Returns an object with get() (number of
        allowed draws or None) and put(n_ok), or None.
        """
        import hashlib
        import json
        d = os.environ.get('BBPOWER_PRIOR_CACHE')
        if not d:
            return None

        def strip(x):
            if isinstance(x, dict):
                return {k: strip(v) for k, v in sorted(x.items())}
            if isinstance(x, (list, tuple)):
                if (len(x) == 3 and str(x[1]).lower() == 'tophat'
                        and isinstance(x[2], (list, tuple))
                        and len(x[2]) == 3):
                    return [x[0], 'tophat', [float(x[2][0]), float(x[2][2])]]
                return [strip(v) for v in x]
            return x

        key = json.dumps(dict(fg_model=strip(self.config['fg_model']),
                              pol_channels=list(self.config['pol_channels']),
                              l_min=self.config['l_min'],
                              l_max=self.config['l_max'],
                              ells=[float(x) for x in self.bpw_l], n=n),
                         sort_keys=True, default=str)
        path = os.path.join(d, 'prior_fraction_' +
                            hashlib.sha1(key.encode()).hexdigest() + '.json')

        class Cache:
            def get(self):
                try:
                    c = json.load(open(path))
                except (OSError, ValueError):
                    return None
                return c['n_ok'] if c.get('key') == key else None

            def put(self, n_ok):
                os.makedirs(d, exist_ok=True)
                tmp = f"{path}.{os.getpid()}.tmp"
                with open(tmp, 'w') as f:
                    json.dump(dict(n_ok=int(n_ok), n=n, key=key), f)
                os.replace(tmp, path)

        return Cache()

    def nautilus_sampler(self):
        """
        Evidence and posterior with nautilus (Lange 2023, MNRAS 525, 3181:
        importance nested sampling with neural-network bounds), same
        likelihood (lnlike) and priors as polychord_sampler (tophat ->
        uniform, Gaussian -> normal). Config block `nautilus` (optional):
        n_live (2000), n_eff (10000, effective posterior samples), seed
        (none), n_networks (4), n_prior (100000, see below). Pool of
        BBPOWER_NPROC processes.

        With moment constraints (moments_positive, moments_hierarchy) the
        prior is the one restricted to the allowed region and normalized
        there, as in PolyChord (points with ln L = -inf are outside its
        prior): nautilus integrates over the whole prior box, so its
        evidence is divided by the allowed fraction f of the prior,
        estimated with n_prior draws from the prior (constraints only, no
        likelihood). Fewer than 100 allowed draws: ValueError ("allowed
        prior fraction too small"), as PolyChord unable to generate its
        live points. With BBPOWER_PRIOR_CACHE (a directory) the fraction is
        computed once per model and reused (_prior_fraction_cache).

        Writes to <output_dir>/nautilus:
          results.npz: log_z (restricted prior), log_z_err (1/sqrt(n_eff)
            and the error of ln f; nautilus gives no error, the scatter
            between seeds is of the order of 1/sqrt(n_eff)), log_z_box
            (whole prior box), log_f, n_eff, n_like, names, points, log_w,
            log_l (weighted posterior)
          equal_weights.txt: equal-weight posterior in the format of
            PolyChord's pch_equal_weights.txt (1, -2 ln L, parameters)
        """
        import time
        from scipy import stats
        from nautilus import Prior, Sampler

        opts = dict(n_live=2000, n_eff=10000, seed=None, n_networks=4,
                    n_prior=100000)
        opts.update(self.config.get('nautilus') or {})
        prior = Prior()
        for name, pr in zip(self.params.p_free_names,
                            self.params.p_free_priors):
            v = [float(x) for x in pr[2]]
            if str(pr[1]).lower() == 'gaussian':
                prior.add_parameter(name, dist=stats.norm(loc=v[0],
                                                          scale=v[1]))
            else:
                prior.add_parameter(name, dist=(v[0], v[2]))

        global _NAUTILUS_LNLIKE, _NAUTILUS_ALLOWED
        _NAUTILUS_LNLIKE = self.lnlike
        _NAUTILUS_ALLOWED = (lambda p: self.moments_allowed(
            self.params.build_params(p)))
        pool = _fork_pool()
        try:
            start = time.time()
            # allowed fraction of the prior (moment constraints)
            log_f, log_f_err = 0., 0.
            if self.moment_constraints():
                n = int(opts['n_prior'])
                cache = self._prior_fraction_cache(n)
                n_ok = cache and cache.get()
                if n_ok is None:
                    rng = np.random.default_rng(opts['seed'])
                    pts = prior.unit_to_physical(
                        rng.random((n, len(self.params.p0))))
                    chunks = np.array_split(pts, 64)
                    n_ok = sum(pool.map(_nautilus_allowed, chunks)
                               if pool is not None else
                               map(_nautilus_allowed, chunks))
                    if cache:
                        cache.put(n_ok)
                # as PolyChord unable to generate its live points: too
                # small an allowed region (and ln f too uncertain)
                if n_ok < 100:
                    raise ValueError(f"allowed prior fraction too small: "
                                     f"{n_ok} of {n} prior samples satisfy "
                                     "the moment constraints")
                f = n_ok / n
                log_f = float(np.log(f))
                log_f_err = float(np.sqrt((1 - f) / (f * n)))
            sampler = Sampler(prior, _nautilus_lnlike, pass_dict=False,
                              n_live=int(opts['n_live']),
                              n_networks=int(opts['n_networks']),
                              pool=pool, seed=opts['seed'])
            sampler.run(n_eff=int(opts['n_eff']), verbose=False)
            elapsed = time.time() - start
        finally:
            if pool is not None:
                pool.close()
                pool.join()

        out = self.get_output('output_dir') + '/nautilus'
        os.makedirs(out, exist_ok=True)
        points, log_w, log_l = sampler.posterior()
        log_z_box = float(sampler.log_z)
        log_z = log_z_box - log_f
        n_eff = float(sampler.n_eff)
        log_z_err = float(np.sqrt(1. / n_eff + log_f_err**2))
        np.savez(out + '/results.npz', log_z=log_z, log_z_err=log_z_err,
                 log_z_box=log_z_box, log_f=log_f, log_f_err=log_f_err,
                 n_eff=n_eff, n_like=int(sampler.n_like),
                 names=self.params.p_free_names, points=points,
                 log_w=log_w, log_l=log_l, time=elapsed)
        # equal-weight samples: about n_eff of them (each point drawn on
        # average boost * w / max(w) times; without boost only
        # sum(w) / max(w) ~ n_eff / 3, too few for the 3 sigma tails)
        w = np.exp(log_w - log_w.max())
        boost = max(1., n_eff / w.sum())
        p_eq, _, l_eq = sampler.posterior(equal_weight=True,
                                          equal_weight_boost=boost)
        np.savetxt(out + '/equal_weights.txt',
                   np.column_stack([np.ones(len(p_eq)), -2 * l_eq, p_eq]))
        print(f"log(Z) = {log_z:.6f} +/- {log_z_err:.6f}  (allowed prior "
              f"fraction: ln f = {log_f:.4f} +/- {log_f_err:.4f}; n_eff "
              f"{n_eff:.0f}, n_like {sampler.n_like}, {elapsed:.1f} s)")
        return sampler

    def polychord_sampler(self):
        import pypolychord
        from pypolychord.settings import PolyChordSettings
        from pypolychord.priors import UniformPrior, GaussianPrior

        ndim = len(self.params.p0)
        nder = 0

        # Log-likelihood compliant with PolyChord's input
        def likelihood(theta):
            return self.lnlike(theta), [0]

        def prior(hypercube):
            prior = []
            for h, pr in zip(hypercube, self.params.p_free_priors):
                if pr[1] == 'Gaussian':
                    prior.append(GaussianPrior(float(pr[2][0]), float(pr[2][1]))(h))  # noqa: E501
                else:
                    prior.append(UniformPrior(float(pr[2][0]), float(pr[2][2]))(h))  # noqa: E501
            return prior

        # Optional dumper function giving run-time read access to
        # the live points, dead points, weights and evidences
        def dumper(live, dead, logweights, logZ, logZerr):  # noqa
            print("Last dead point:", dead[-1])

        settings = PolyChordSettings(ndim, nder)
        settings.base_dir = self.get_output('output_dir')+'/polychord'
        settings.file_root = 'pch'
        settings.nlive = self.config['nlive']
        settings.num_repeats = self.config['nrepeat']
        settings.do_clustering = False  # Assume unimodal posterior
        settings.boost_posterior = 10   # Increase number of posterior samples
        settings.nprior = 200           # Draw nprior initial prior samples
        settings.maximise = True        # Maximize posterior at the end
        settings.read_resume = False    # Read from resume file of earlier run
        settings.feedback = 2           # Verbosity {0,1,2,3}

        output = pypolychord.run_polychord(likelihood, ndim, nder, settings, 
                                           prior, dumper)

        return output

    def minimizer(self):
        """
        Find maximum likelihood
        """
        from scipy.optimize import minimize

        def chi2(par):
            c2 = -2*self.lnprob(par)
            return c2

        res = minimize(chi2, self.params.p0,
                       method="Powell")
        return res.x

    def fisher(self):
        """
        Evaluate Fisher matrix
        """
        import numdifftools as nd
        from scipy.optimize import minimize

        def chi2(par):
            c2 = -2*self.lnprob(par)
            return c2

        res = minimize(chi2, self.params.p0,
                       method="Powell")

        def lnprobd(p):
            l = self.lnprob(p)  # noqa
            if l == -np.inf:
                l = -1E100  # noqa
            return l

        fisher = - nd.Hessian(lnprobd)(res.x)
        return res.x, fisher

    def singlepoint(self):
        """
        Evaluate at a single point
        """
        chi2 = -2*self.lnprob(self.params.p0)
        return chi2

    def timing(self, n_eval=300):
        """
        Evaluate n times and benchmark
        """
        import time
        start = time.time()
        for i in range(n_eval):
            self.lnprob(self.params.p0)
        end = time.time()

        return end-start, (end-start)/n_eval
    
    def predicted_spectra(self, at_min=True, save_npz=True):
        """
        Evaluates model at a the maximum likelihood and
        writes predicted spectra into a numpy array
        with shape (nbpws, nmaps, nmaps).
        """
        if at_min:
            sampler = self.minimizer()
            p = np.array(sampler)
        else:
            p = self.params.p0
        pars = self.params.build_params(p)
        model_cls = self.model(pars)
        if self.config['bands'] == 'all':
            tr_names = sorted(list(self.s.tracers.keys()))
        else:
            tr_names = self.config['bands']
        if save_npz:
            np.savez(self.get_output('output_dir')+'/cells_model.npz',
                     tracers=tr_names, 
                     ls=self.ell_b,
                     dls=model_cls)
            return
        s = sacc.Sacc()
        for it, tn in enumerate(tr_names):
            t = self.s.tracers[tn]
            s.add_tracer('NuMap', tn, quantity='cmb_polarization',
                         spin=2, nu=t.nu, bandpass=t.bandpass,
                         ell=t.ell, beam=t.beam, nu_unit='GHz',
                         map_unit='uK_CMB')
        for b1, b2, p1, p2, m1, m2, ind in self._freq_pol_iterator():
            cl = model_cls[:, m1, m2]
            t1 = tr_names[b1]
            t2 = tr_names[b2]
            pol1 = self.pols[p1].lower()
            pol2 = self.pols[p2].lower()
            cltyp = f'cl_{pol1}{pol2}'
            win = sacc.BandpowerWindow(self.bpw_l, self.windows[ind].T)
            s.add_ell_cl(cltyp, t1, t2, self.ell_b, cl, window=win)
        s.add_covariance(self.bbcovar)
        s.save_fits(self.get_output('output_dir')+'/cells_model.fits',
                    overwrite=True)
        return

    def run(self):
        from shutil import copyfile
        copyfile(self.get_input('config'), self.get_output('config_copy'))
        self.setup_compsep()
        if self.config.get('sampler') == 'emcee':
            sampler, timing = self.emcee_sampler()
            chi2 = -2*self.lnprob(self.minimizer())
            np.savez(self.get_output('output_dir')+'/emcee.npz',
                     chain=sampler.chain,
                     names=self.params.p_free_names,
                     time=timing,
                     chi2=chi2,
                     ndof=len(self.bbcovar))
            print("Finished sampling", timing)
        elif self.config.get('sampler') == 'polychord':
            sampler = self.polychord_sampler()
            print("Finished sampling")
        elif self.config.get('sampler') == 'nautilus':
            sampler = self.nautilus_sampler()
            print("Finished sampling")
        elif self.config.get('sampler') == 'fisher':
            p0, fisher = self.fisher()
            cov = np.linalg.inv(fisher)
            for i, (n, p) in enumerate(zip(self.params.p_free_names, p0)):
                print(n+" = %.3lE +- %.3lE" % (p, np.sqrt(cov[i, i])))
            np.savez(self.get_output('output_dir')+'/fisher.npz',
                     params=p0, fisher=fisher,
                     names=self.params.p_free_names)
        elif self.config.get('sampler') == 'maximum_likelihood':
            sampler = self.minimizer()
            chi2 = -2*self.lnprob(sampler)
            np.savez(self.get_output('output_dir')+'/chi2.npz',
                     params=sampler,
                     names=self.params.p_free_names,
                     chi2=chi2, ndof=len(self.bbcovar))
            print("Best fit:")
            for n, p in zip(self.params.p_free_names, sampler):
                print(n+" = %.3lE" % p)
            print("Chi2: %.3lE" % chi2)
        elif self.config.get('sampler') == 'single_point':
            sampler = self.singlepoint()
            np.savez(self.get_output('output_dir')+'/single_point.npz',
                     chi2=sampler, ndof=len(self.bbcovar),
                     names=self.params.p_free_names)
            print("Chi2:", sampler, len(self.bbcovar))
        elif self.config.get('sampler') == 'timing':
            sampler = self.timing()
            np.savez(self.get_output('output_dir')+'/timing.npz',
                     timing=sampler[1],
                     names=self.params.p_free_names)
            print("Total time:", sampler[0])
            print("Time per eval:", sampler[1])
        elif self.config.get('sampler') == 'predicted_spectra':
            at_min = self.config.get('predict_at_minimum', True)
            save_npz = not self.config.get('predict_to_sacc', False)
            sampler = self.predicted_spectra(at_min=at_min, save_npz=save_npz)
            print("Predicted spectra saved")
        else:
            raise ValueError("Unknown sampler")

        return


if __name__ == '__main__':
    cls = PipelineStage.main()
