import numpy as np


class ParameterManager(object):

    def _add_parameter(self, p_name, p):
        # If fixed parameter, just add its name and value
        if p[1] == 'fixed':
            self.p_fixed.append((p_name, float(p[2][0])))
            return  # Then move on

        # Otherwise it's free
        # Check for duplicate names
        if p_name in self.p_free_names:
            raise KeyError("You have two parameters with the same name")
        # Add name and prior to list
        self.p_free_names.append(p_name)
        self.p_free_priors.append(p)
        # Add fiducial value to initial vector
        if np.char.lower(p[1]) == 'tophat':
            p0 = float(p[2][1])
        elif np.char.lower(p[1]) == 'gaussian':
            p0 = float(p[2][0])
        else:
            raise ValueError("Unknown prior type %s" % p[1])
        self.p0.append(p0)

    def _add_parameters(self, params):
        for p_name in sorted(params.keys()):
            p = params[p_name]
            self._add_parameter(p_name, p)

    def get_component_names(self, config):
        comps = []
        for c_name in config['fg_model'].keys():
            if c_name.startswith('component_'):
                comps.append(c_name)
        return sorted(comps)

    def __init__(self, config):
        self.p_free_names = []
        self.p_free_priors = []
        self.p_fixed = []
        self.p0 = []

        # CMB parameters
        d = config.get('cmb_model')
        if d:
            self._add_parameters(d['params'])

        # Loop through FG components
        comp_names = self.get_component_names(config)
        for c_name in comp_names:
            c = config['fg_model'][c_name]
            for tag in ['sed_parameters', 'cross', 'decorr']:
                d = c.get(tag)
                if d:
                    self._add_parameters(d)
            dc = c.get('cl_parameters')
            if dc:  # Power spectra
                for cl_name, d in dc.items():
                    p1, p2 = cl_name
                    # Add parameters only if we're using both
                    # polarization channels
                    if ((p1 in config['pol_channels']) and
                            (p2 in config['pol_channels'])):
                        self._add_parameters(d)

            dm = c.get('moments')
            if dm and config['fg_model'].get('use_moments'):  # Moments
                self._add_parameters(dm)

        # Loop through different systematics
        if 'systematics' in config.keys():
            cnf_sys = config['systematics']
            # Bandpasses
            if 'bandpasses' in cnf_sys.keys():
                cnf_bps = cnf_sys['bandpasses']
                i_bps = 1
                while 'bandpass_%d' % i_bps in cnf_bps:
                    if cnf_bps['bandpass_%d' % i_bps].get('parameters'):
                        self._add_parameters(cnf_bps['bandpass_%d' % i_bps]['parameters'])  # noqa
                    i_bps += 1

        self.p0 = np.array(self.p0)
        # priors as (is_gaussian, a, b): (mean, sigma) or (low, high), so
        # that lnprior does not parse the prior names at every call
        self._priors = []
        for pr in self.p_free_priors:
            gauss = str(pr[1]).lower() == 'gaussian'
            self._priors.append((gauss, float(pr[2][0]),
                                 float(pr[2][1] if gauss else pr[2][2])))

    def build_params(self, par):
        params = dict(self.p_fixed)
        params.update(dict(zip(self.p_free_names, par)))
        return params

    def lnprior(self, par):
        lnp = 0
        for p, (gauss, a, b) in zip(par, self._priors):
            if gauss:  # Gaussian prior
                lnp += -0.5 * ((p - a)/b)**2
            else:  # Only other option is top-hat
                if not (a <= p <= b):
                    return -np.inf
        return lnp
