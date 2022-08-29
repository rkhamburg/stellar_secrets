import bilby

def condition_func(parameters):
    converted_parameters = parameters.copy()
    converted_parameters['lb'] = parameters['l0'] - parameters['l1']
    return converted_parameters

def condition_func2(parameters):
    converted_parameters = parameters.copy()
    converted_parameters['lb'] = parameters['l0'] - parameters['l1']
    converted_parameters['lb2'] = parameters['ml0'] - parameters['ml1']
    return converted_parameters

def make_prior(ldist, zdist='SFR'):
    # Set up prior dict
    if ldist == 'BPL':
        priors = bilby.core.prior.PriorDict(conversion_function=condition_func)
    else:
        priors = bilby.core.prior.PriorDict()

    # Set up default prior distributions
    priors['z0'] = bilby.prior.Uniform(minimum=-1., maximum=2., name='local rate density', latex_label=r'$\rho_0$')
    priors['l0'] = bilby.prior.Uniform(minimum=-3., maximum=0., name='luminosity low index', latex_label=r'$\alpha_L$')

    # Add cut-off luminosity
    if ldist == 'CPL':
        priors['l2'] = bilby.prior.Uniform(minimum=49, maximum=55, name='cutoff luminosity', latex_label=r'$L_c$')

    # Add break luminosity and high-luminosity power law
    if ldist == 'BPL':
        priors['l1'] = bilby.prior.Uniform(minimum=-4., maximum=0., name='luminosity high index', latex_label=r'$\beta_L$')
        priors['l2'] = bilby.prior.Uniform(minimum=49, maximum=55, name='break luminosity', latex_label=r'$L_*$')
        priors['lb'] = bilby.prior.Constraint(minimum=0, maximum=4)

    # Add wanderman and piran redshift rate
    if zdist == 'WP':
        priors['z1'] = bilby.prior.Uniform(minimum=0, maximum=5, name='redshift low index', latex_label=r'$\alpha_z$')
        priors['z2'] = bilby.prior.Uniform(minimum=-5, maximum=0, name='redshift high index', latex_label=r'$\beta_z$')
        priors['z3'] = bilby.prior.Uniform(minimum=0., maximum=10, name='break redshift', latex_label=r'$z_*$')

    return priors


def make_prior_merger(ldist):
    # Set up prior dict
    if ldist == 'BPL' or ldist == 'DBPL':
        priors = bilby.core.prior.PriorDict(conversion_function=condition_func)
    else:
        priors = bilby.core.prior.PriorDict()

    # Set up default prior distributions
    priors['z0'] = bilby.prior.Uniform(minimum=-1., maximum=3., name='local rate density', latex_label=r'$\rho_0$')
    priors['l0'] = bilby.prior.Uniform(minimum=-3., maximum=0., name='luminosity low index', latex_label=r'$\alpha_L$')


    # Add cut-off luminosity
    if ldist == 'CPL':
        priors['l2'] = bilby.prior.Uniform(minimum=49, maximum=55, name='cutoff luminosity', latex_label=r'$L_c$')

    # Add break luminosity and high-luminosity power law
    elif ldist == 'BPL':
        priors['l1'] = bilby.prior.Uniform(minimum=-4., maximum=0., name='luminosity high index', latex_label=r'$\beta_L$')
        priors['l2'] = bilby.prior.Uniform(minimum=49, maximum=55, name='break luminosity', latex_label=r'$L_*$')
        priors['lb'] = bilby.prior.Constraint(minimum=0, maximum=4)

    elif ldist == 'DBPL':
        priors['l1'] = bilby.prior.Uniform(minimum=-4., maximum=0., name='luminosity high index', latex_label=r'$\beta_L$')
        priors['l2'] = bilby.prior.Uniform(minimum=49, maximum=55, name='break luminosity', latex_label=r'$L_*$')
        priors['l3'] = bilby.prior.Uniform(minimum=-1., maximum=1.)
        priors['lb'] = bilby.prior.Constraint(minimum=0., maximum=4.)

    return priors


def make_prior_joint(ldist, zdist, ldist2):
    # Set up prior dict
    if ldist == 'BPL':
        priors = bilby.core.prior.PriorDict(conversion_function=condition_func)
    elif ldist2 == 'BPL':
        priors = bilby.core.prior.PriorDict(conversion_function=condition_func2)
    else:
        priors = bilby.core.prior.PriorDict()

    # Set up default prior distributions
    priors['z0'] = bilby.prior.Uniform(minimum=-1., maximum=2., name='local rate density', latex_label=r'$\rho_0$')
    priors['l0'] = bilby.prior.Uniform(minimum=-3., maximum=0., name='luminosity low index', latex_label=r'$\alpha_L$')

    # Add cut-off luminosity
    if ldist == 'CPL':
        priors['l2'] = bilby.prior.Uniform(minimum=49, maximum=55, name='cutoff luminosity', latex_label=r'$L_c$')

    # Add break luminosity and high-luminosity power law
    elif ldist == 'BPL':
        priors['l1'] = bilby.prior.Uniform(minimum=-4., maximum=0., name='luminosity high index', latex_label=r'$\beta_L$')
        priors['l2'] = bilby.prior.Uniform(minimum=49, maximum=55, name='break luminosity', latex_label=r'$L_*$')
        priors['lb'] = bilby.prior.Constraint(minimum=0, maximum=4)

    # Add wanderman and piran redshift rate
    if zdist == 'WP':
        priors['z1'] = bilby.prior.Uniform(minimum=0, maximum=5, name='redshift low index', latex_label=r'$\alpha_z$')
        priors['z2'] = bilby.prior.Uniform(minimum=-5, maximum=0, name='redshift high index', latex_label=r'$\beta_z$')
        priors['z3'] = bilby.prior.Uniform(minimum=0., maximum=10, name='break redshift', latex_label=r'$z_*$')

    # Defaults for mergers
    priors['mz0'] = bilby.prior.Uniform(minimum=-1., maximum=3., name='local rate density (merger)', latex_label=r'$\rho_0^M$')
    priors['ml0'] = bilby.prior.Uniform(minimum=-3., maximum=0., name='luminosity low index', latex_label=r'$\alpha_L^M$')

    if ldist2 == 'CPL':
        priors['ml2'] = bilby.prior.Uniform(minimum=47, maximum=54, name='cutoff luminosity', latex_label=r'$L_c$')

    elif ldist2 == 'BPL':
        priors['ml1'] = bilby.prior.Uniform(minimum=-4., maximum=0., name='luminosity high index', latex_label=r'$\beta_L$')
        priors['ml2'] = bilby.prior.Uniform(minimum=47, maximum=54, name='break luminosity', latex_label=r'$L_*$')
        priors['lb2'] = bilby.prior.Constraint(minimum=0, maximum=4)

    return priors
