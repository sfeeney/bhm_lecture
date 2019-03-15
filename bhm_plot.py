import numpy as np
import matplotlib.pyplot as mp
import daft
mp.rc("font", family="serif", size=11.5)
mp.rc("text", usetex=True)
mp.rcParams['text.latex.preamble']=[r"\usepackage{bm}"]
# colours
s_color = {"ec": "#f89406"}
p_color = {"ec": "#46a546"}

# standard version of Cepheid period-luminosity relation example

# positions
data_level, like_level, obj_level, \
	obj_prior_level, pop_level, pop_prior_level, \
	pop_prior_par_level = np.arange(1, 8) - 0.25

# latex
def pr(vble):
	return r'{\rm P}(' + vble + ')'
def objectify(vble, n_inds=1):
	if n_inds == 2:
		return vble + r'_{ij}'
	else:
		return vble + r'_i'

mu_true = objectify(r'\mu')
mu_prior_mean = r'\bar{\mu}'
mu_prior_sig = r'\sigma_\mu'
mu_obs = objectify(r'\hat{\mu}')
mu_obs_sig = r'\sigma_{' + objectify(r'\hat{\mu}') + r'}'
abs_mag = r'M^*'
abs_mag_prior_mean = r'\bar{M^*}'
abs_mag_prior_sig = r'\sigma_{M^*}'
slope = r's'
slope_prior_mean = r'\bar{s}'
slope_prior_sig = r'\sigma_s'
m_true = objectify(r'm', 2)
m_obs = objectify(r'\hat{m}', 2)
m_obs_sig = r'\sigma_{' + objectify(r'\hat{m}', 2) + r'}'
m_sig_int = r'\sigma_{\rm int}'
log_p = objectify(r'\log p', 2)

# create figure
pgm = daft.PGM([5.25, 7.25], origin=[-0.5, 0.0], observed_style="inner")

# population-prior parameters
pgm.add_node(daft.Node('abs_mag_prior_mean', '$' + abs_mag_prior_mean + '$', 1.5, \
					   pop_prior_par_level, fixed=True))
pgm.add_node(daft.Node('abs_mag_prior_sig', '$' + abs_mag_prior_sig + '$', 2.5, \
					   pop_prior_par_level, fixed=True))
pgm.add_node(daft.Node('slope_prior_mean', '$' + slope_prior_mean + '$', 3.5, \
					   pop_prior_par_level, fixed=True))
pgm.add_node(daft.Node('slope_prior_sig', '$' + slope_prior_sig + '$', 4.5, \
					   pop_prior_par_level, fixed=True))

# population priors
pgm.add_node(daft.Node('pr_abs_mag', '$' + \
					   pr(abs_mag + '|' + abs_mag_prior_mean + ',' + \
					   	  abs_mag_prior_sig) + '$', 2.0, pop_prior_level, \
					   aspect = 3.0, plot_params=s_color, \
					   shape='rectangle'))
pgm.add_node(daft.Node('pr_slope', '$' + \
					   pr(slope + '|' + slope_prior_mean + ',' + \
					   	  slope_prior_sig) + '$', 4.0, pop_prior_level, \
					   aspect = 1.8, plot_params=s_color, \
					   shape='rectangle'))

# population-level parameters
pgm.add_node(daft.Node('mu_prior_mean', '$' + mu_prior_mean + '$', 0.0, \
					   pop_level, fixed=True))
pgm.add_node(daft.Node('mu_prior_sig', '$' + mu_prior_sig + '$', 1.0, \
					   pop_level, fixed=True))
pgm.add_node(daft.Node('abs_mag', '$' + abs_mag + '$', 2.0, pop_level))
pgm.add_node(daft.Node('m_sig_int', '$' + m_sig_int + '$', 3.0, pop_level, \
					   fixed=True))
pgm.add_node(daft.Node('slope', '$' + slope + '$', 4.0, pop_level))

# object priors
pgm.add_node(daft.Node('pr_mu_true_i', \
					   '$' + pr(mu_true + '|' + mu_prior_mean + ',' + \
								mu_prior_sig) + '$', \
					   1.0, obj_prior_level, aspect = 2.1, \
					   plot_params=s_color, shape='rectangle'))
pgm.add_node(daft.Node('pr_m_true_ij', \
					   '$' + pr(m_true + '|' + \
					   			mu_true + ',' + \
								abs_mag + ',' + slope + ',' + \
								log_p + ',' + \
								m_sig_int) + '$', \
					   3.0, obj_prior_level, aspect = 5.1, \
					   plot_params=s_color, shape='rectangle'))

# object-level parameters
pgm.add_node(daft.Node('mu_true_i', '$' + mu_true + '$', \
					   1.0, obj_level))
pgm.add_node(daft.Node('m_true_ij', '$' + m_true + '$', \
					   3.0, obj_level))
pgm.add_node(daft.Node('lp_ij', '', 4.0, \
					   obj_level, fixed=True))

# likelihoods
pgm.add_node(daft.Node('pr_mu_obs_i', \
					   '$' + pr(mu_obs + '|' + mu_true + ',' + mu_obs_sig) + '$', \
					   1.0, like_level, aspect=2.4, plot_params=s_color, \
					   shape='rectangle'))
pgm.add_node(daft.Node('pr_m_obs_ij', \
					   '$' + pr(m_obs + '|' + m_true + ',' + m_obs_sig) + '$', \
					   3.0, like_level, aspect=3.0, plot_params=s_color, \
					   shape='rectangle'))

# observables
pgm.add_node(daft.Node('mu_obs_i', '$' + mu_obs + '$', 1.0, data_level, \
					   observed=True))
pgm.add_node(daft.Node('mu_obs_sig_i', '', 0.0, data_level, \
					   fixed=True))
pgm.add_node(daft.Node('m_obs_ij', '$' + m_obs + '$', 3.0, data_level, \
					   observed=True))
pgm.add_node(daft.Node('m_obs_sig_ij', '', 4.0, data_level, \
					   fixed=True))

# edges
pgm.add_edge('abs_mag_prior_mean', 'pr_abs_mag')
pgm.add_edge('abs_mag_prior_sig', 'pr_abs_mag')
pgm.add_edge('slope_prior_mean', 'pr_slope')
pgm.add_edge('slope_prior_sig', 'pr_slope')
pgm.add_edge('pr_abs_mag', 'abs_mag')
pgm.add_edge('pr_abs_mag', 'abs_mag')
pgm.add_edge('pr_slope', 'slope')
pgm.add_edge('pr_slope', 'slope')
pgm.add_edge('mu_prior_mean', 'pr_mu_true_i')
pgm.add_edge('mu_prior_sig', 'pr_mu_true_i')
pgm.add_edge('abs_mag', 'pr_m_true_ij')
pgm.add_edge('m_sig_int', 'pr_m_true_ij')
pgm.add_edge('slope', 'pr_m_true_ij')
pgm.add_edge('mu_true_i', 'pr_m_true_ij')
pgm.add_edge('lp_ij', 'pr_m_true_ij')
pgm.add_edge('pr_mu_true_i', 'mu_true_i')
pgm.add_edge('pr_m_true_ij', 'm_true_ij')
pgm.add_edge('mu_true_i', 'pr_mu_obs_i')
pgm.add_edge('m_true_ij', 'pr_m_obs_ij')
pgm.add_edge('pr_mu_obs_i', 'mu_obs_i')
pgm.add_edge('mu_obs_sig_i', 'pr_mu_obs_i')
pgm.add_edge('pr_m_obs_ij', 'm_obs_ij')
pgm.add_edge('m_obs_sig_ij', 'pr_m_obs_ij')

# object plate
pgm.add_plate(daft.Plate([1.65, data_level - 0.5, 2.7, \
						  pop_level - data_level], \
						  label=r"$1 \le j \le n_{\rm star}$", \
						  shift=-0.0, rect_params={"ec": "r"}, \
						  label_offset=(2, 2)))
pgm.add_plate(daft.Plate([-0.25, data_level - 0.6, 4.7, \
						  pop_level - data_level + 0.2], \
						  label=r"$1 \le i \le n_{\rm gal}$", \
						  shift=-0.0, rect_params={"ec": "r"}, \
						  label_offset=(2, 2)))

# render and save
pgm.render()
pgm.figure.text(4.5 / 5.25, (obj_level - 0.25) / 7.25, \
				'$' + log_p + '$', ha='center')
pgm.figure.text(0.5 / 5.25, (data_level - 0.25) / 7.25, \
				'$' + mu_obs_sig + '$', ha='center')
pgm.figure.text(4.5 / 5.25, (data_level - 0.25) / 7.25, \
				'$' + m_obs_sig + '$', ha='center')
pgm.figure.savefig('bhm_plot.pdf')

# no PDFs version
# create figure
pgm = daft.PGM([5.25, 7.25], origin=[-0.5, 0.0], observed_style="inner")

# population-prior parameters
pgm.add_node(daft.Node('abs_mag_prior_mean', '$' + abs_mag_prior_mean + '$', 1.5, \
					   pop_prior_par_level, fixed=True))
pgm.add_node(daft.Node('abs_mag_prior_sig', '$' + abs_mag_prior_sig + '$', 2.5, \
					   pop_prior_par_level, fixed=True))
pgm.add_node(daft.Node('slope_prior_mean', '$' + slope_prior_mean + '$', 3.5, \
					   pop_prior_par_level, fixed=True))
pgm.add_node(daft.Node('slope_prior_sig', '$' + slope_prior_sig + '$', 4.5, \
					   pop_prior_par_level, fixed=True))

# population-level parameters
pgm.add_node(daft.Node('mu_prior_mean', '$' + mu_prior_mean + '$', 0.0, \
					   pop_level, fixed=True))
pgm.add_node(daft.Node('mu_prior_sig', '$' + mu_prior_sig + '$', 1.0, \
					   pop_level, fixed=True))
pgm.add_node(daft.Node('abs_mag', '$' + abs_mag + '$', 2.0, pop_level))
pgm.add_node(daft.Node('m_sig_int', '$' + m_sig_int + '$', 3.0, pop_level, \
					   fixed=True))
pgm.add_node(daft.Node('slope', '$' + slope + '$', 4.0, pop_level))

# object-level parameters
pgm.add_node(daft.Node('mu_true_i', '$' + mu_true + '$', \
					   1.0, obj_level))
pgm.add_node(daft.Node('m_true_ij', '$' + m_true + '$', \
					   3.0, obj_level))
pgm.add_node(daft.Node('lp_ij', '', 4.0, \
					   obj_level, fixed=True))

# observables
pgm.add_node(daft.Node('mu_obs_i', '$' + mu_obs + '$', 1.0, data_level, \
					   observed=True))
pgm.add_node(daft.Node('mu_obs_sig_i', '$' + mu_obs_sig + '$', 0.0, data_level, \
					   fixed=True))
pgm.add_node(daft.Node('m_obs_ij', '$' + m_obs + '$', 3.0, data_level, \
					   observed=True))
pgm.add_node(daft.Node('m_obs_sig_ij', '$' + m_obs_sig + '$', 4.0, data_level, \
					   fixed=True))

# edges
pgm.add_edge('abs_mag_prior_mean', 'abs_mag')
pgm.add_edge('abs_mag_prior_sig', 'abs_mag')
pgm.add_edge('slope_prior_mean', 'slope')
pgm.add_edge('slope_prior_sig', 'slope')
pgm.add_edge('mu_prior_mean', 'mu_true_i')
pgm.add_edge('mu_prior_sig', 'mu_true_i')
pgm.add_edge('abs_mag', 'm_true_ij')
pgm.add_edge('m_sig_int', 'm_true_ij')
pgm.add_edge('slope', 'm_true_ij')
pgm.add_edge('mu_true_i', 'm_true_ij')
pgm.add_edge('lp_ij', 'm_true_ij')
pgm.add_edge('mu_true_i', 'mu_obs_i')
pgm.add_edge('m_true_ij', 'm_obs_ij')
pgm.add_edge('mu_obs_sig_i', 'mu_obs_i')
pgm.add_edge('m_obs_sig_ij', 'm_obs_ij')

# object plate
pgm.add_plate(daft.Plate([1.65, data_level - 0.5, 2.7, \
						  pop_level - data_level], \
						  label=r"$1 \le j \le n_{\rm star}$", \
						  shift=-0.0, rect_params={"ec": "r"}, \
						  label_offset=(2, 2)))
pgm.add_plate(daft.Plate([-0.25, data_level - 0.6, 4.7, \
						  pop_level - data_level + 0.2], \
						  label=r"$1 \le i \le n_{\rm gal}$", \
						  shift=-0.0, rect_params={"ec": "r"}, \
						  label_offset=(2, 2)))

# render and save
pgm.render()
pgm.figure.text(4.25 / 5.25, (obj_level - 0.25) / 7.25, '$' + log_p + '$')
pgm.figure.savefig('bhm_plot_no_pdfs.pdf')


# very simple Gaussian version
# positions
data_level, obj_level, pop_level, pop_prior_par_level = np.arange(1, 5) - 0.25

# latex
m_prior_mean = r'\bar{M}'
m_prior_sig = r'\sigma_{M}'
m_true = objectify(r'M')
m_obs = objectify(r'\hat{M}')
m_obs_sig = r'\sigma_{' + objectify(r'\hat{M}') + r'}'

# create figure
pgm = daft.PGM([3.5, 4.25], origin=[0.25, 0.0], observed_style="inner")

# population-prior parameters
pgm.add_node(daft.Node('prior_info', '$I$ (prior info)', 2.0, \
					   pop_prior_par_level, fixed=True))

# population-level parameters
pgm.add_node(daft.Node('m_prior_mean', '$' + m_prior_mean + '$', 1.0, \
					   pop_level))
pgm.add_node(daft.Node('m_prior_sig', '$' + m_prior_sig + '$', 3.0, \
					   pop_level))

# object-level parameters
pgm.add_node(daft.Node('m_true_i', '$' + m_true + '$', \
					   2.0, obj_level))

# observables
pgm.add_node(daft.Node('m_obs_i', '$' + m_obs + '$', 2.0, data_level, \
					   observed=True))
pgm.add_node(daft.Node('m_obs_sig_i', '$' + m_obs_sig + '$', 1.0, data_level, \
					   fixed=True))

# edges
pgm.add_edge('prior_info', 'm_prior_mean')
pgm.add_edge('prior_info', 'm_prior_sig')
pgm.add_edge('m_prior_mean', 'm_true_i')
pgm.add_edge('m_prior_sig', 'm_true_i')
pgm.add_edge('m_true_i', 'm_obs_i')
pgm.add_edge('m_obs_sig_i', 'm_obs_i')

# object plate
pgm.add_plate(daft.Plate([0.5, data_level - 0.5, 3.0, \
						  pop_level - data_level], \
						  label=r"$1 \le i \le n_{\rm star}$", \
						  shift=-0.0, rect_params={"ec": "r"}, \
						  label_offset=(2, 2)))

# render and save
pgm.render()
pgm.figure.savefig('bhm_plot_1dg.pdf')


# Gaussian version with multiple classes
# latex
def classify(vble):
	return vble + r'_k'
m_prior_mean = r'\bar{' + classify('M') + r'}'
m_prior_sig = r'\sigma_{' + classify('M') + r'}'
class_prob = classify('p')
m_true = objectify(r'M')
class_true = objectify(r'\kappa')
m_obs = objectify(r'\hat{M}')
m_obs_sig = r'\sigma_{' + objectify(r'\hat{M}') + r'}'

# create figure
pgm = daft.PGM([3.5, 4.5], origin=[0.25, 0.0], observed_style="inner")

# population-prior parameters
pgm.add_node(daft.Node('prior_info', '$I$', 2.0, \
					   pop_prior_par_level, fixed=True))

# population-level parameters
pgm.add_node(daft.Node('m_prior_mean', '$' + m_prior_mean + '$', 1.0, \
					   pop_level))
pgm.add_node(daft.Node('m_prior_sig', '$' + m_prior_sig + '$', 2.0, \
					   pop_level))
pgm.add_node(daft.Node('class_prob', '$' + class_prob + '$', 3.0, \
					   pop_level))

# object-level parameters
pgm.add_node(daft.Node('m_true_i', '$' + m_true + '$', \
					   2.0, obj_level))
pgm.add_node(daft.Node('class_i', '$' + class_true + '$', \
					   3.0, obj_level))

# observables
pgm.add_node(daft.Node('m_obs_i', '$' + m_obs + '$', 2.0, data_level, \
					   observed=True))
pgm.add_node(daft.Node('m_obs_sig_i', '$' + m_obs_sig + '$', 1.0, data_level, \
					   fixed=True))

# edges
pgm.add_edge('prior_info', 'm_prior_mean')
pgm.add_edge('prior_info', 'm_prior_sig')
pgm.add_edge('prior_info', 'class_prob')
pgm.add_edge('m_prior_mean', 'm_true_i')
pgm.add_edge('m_prior_sig', 'm_true_i')
pgm.add_edge('class_i', 'm_true_i')
pgm.add_edge('class_prob', 'class_i')
pgm.add_edge('m_true_i', 'm_obs_i')
pgm.add_edge('m_obs_sig_i', 'm_obs_i')

# object plate
pgm.add_plate(daft.Plate([0.5, data_level - 0.5, 3.0, \
						  pop_level - data_level - 0.02], \
						  label=r"$i < n_{\rm star}$", \
						  shift=-0.0, rect_params={"ec": "r"}, \
						  label_offset=(2, 2)))
pgm.add_plate(daft.Plate([0.5, pop_level - 0.48, 3.0, \
						  pop_prior_par_level - pop_level + 1], \
						  label=r"$k < n_{\rm cls}$", \
						  shift=-0.0, rect_params={"ec": "r"}, \
						  label_offset=(2, 2)))

# render and save
pgm.render()
pgm.figure.savefig('bhm_plot_1dg_class.pdf')


# Gaussian version with multiple classes
# latex
m_obs = objectify(r'\hat{M}', 2)
m_obs_sig = r'\sigma_{' + objectify(r'\hat{M}', 2) + r'}'

# create figure
pgm = daft.PGM([3.5, 4.5], origin=[0.25, 0.0], observed_style="inner")

# population-prior parameters
pgm.add_node(daft.Node('prior_info', '$I$', 2.0, \
					   pop_prior_par_level, fixed=True))

# population-level parameters
pgm.add_node(daft.Node('m_prior_mean', '$' + m_prior_mean + '$', 1.0, \
					   pop_level))
pgm.add_node(daft.Node('m_prior_sig', '$' + m_prior_sig + '$', 2.0, \
					   pop_level))
pgm.add_node(daft.Node('class_prob', '$' + class_prob + '$', 3.0, \
					   pop_level))

# object-level parameters
pgm.add_node(daft.Node('m_true_i', '$' + m_true + '$', \
					   2.0, obj_level))
pgm.add_node(daft.Node('class_i', '$' + class_true + '$', \
					   3.0, obj_level))

# observables
pgm.add_node(daft.Node('m_obs_ij', '$' + m_obs + '$', 2.0, data_level, \
					   observed=True))
pgm.add_node(daft.Node('m_obs_sig_ij', '$' + m_obs_sig + '$', 1.0, data_level, \
					   fixed=True))

# edges
pgm.add_edge('prior_info', 'm_prior_mean')
pgm.add_edge('prior_info', 'm_prior_sig')
pgm.add_edge('prior_info', 'class_prob')
pgm.add_edge('m_prior_mean', 'm_true_i')
pgm.add_edge('m_prior_sig', 'm_true_i')
pgm.add_edge('class_i', 'm_true_i')
pgm.add_edge('class_prob', 'class_i')
pgm.add_edge('m_true_i', 'm_obs_ij')
pgm.add_edge('m_obs_sig_ij', 'm_obs_ij')

# object plate
pgm.add_plate(daft.Plate([0.5, data_level - 0.5, 3.0, \
						  pop_level - data_level - 0.02], \
						  label=r"$i < n_{\rm star}$", \
						  shift=-0.0, rect_params={"ec": "r"}, \
						  label_offset=(127, 2)))
pgm.add_plate(daft.Plate([0.54, data_level - 0.46, 1.96, 0.96], \
						  label=r"$j < n_{\rm obs}$", \
						  shift=-0.0, rect_params={"ec": "r"}, \
						  label_offset=(2, 2)))
pgm.add_plate(daft.Plate([0.5, pop_level - 0.48, 3.0, \
						  pop_prior_par_level - pop_level + 1], \
						  label=r"$k < n_{\rm cls}$", \
						  shift=-0.0, rect_params={"ec": "r"}, \
						  label_offset=(2, 2)))

# render and save
pgm.render()
pgm.figure.savefig('bhm_plot_1dg_class_obs.pdf')


# Gaussian version with multiple classes
# positions
data_level, like_level, obj_level, \
	obj_prior_level, pop_level, pop_prior_level, \
	pop_prior_par_level = np.arange(1, 8) - 0.25

# latex
m_prior_mean_k = r'\bar{M}_{k=' + class_true + '}'
m_prior_sig_k = r'\sigma_{M_{k=' + class_true + '}}'
class_probs = r'\bm{p}'

# create figure
pgm = daft.PGM([3.5, 7.5], origin=[0.25, 0.0], observed_style="inner")

# population-prior parameters
pgm.add_node(daft.Node('prior_info', '$I$', 2.0, \
					   pop_prior_par_level, fixed=True))

# population priors
pgm.add_node(daft.Node('pr_m_prior_mean', '$' + \
					   pr(m_prior_mean + '|I') + '$', 1.0, pop_prior_level, \
					   aspect = 1.7, plot_params=s_color, \
					   shape='rectangle'))
pgm.add_node(daft.Node('pr_m_prior_sig', '$' + \
					   pr(m_prior_sig + '|I') + '$', 2.0, pop_prior_level, \
					   aspect = 1.7, plot_params=s_color, \
					   shape='rectangle'))
pgm.add_node(daft.Node('pr_class_prob', '$' + \
					   pr(class_prob + '|I') + '$', 3.0, pop_prior_level, \
					   aspect = 1.7, plot_params=s_color, \
					   shape='rectangle'))

# population-level parameters
pgm.add_node(daft.Node('m_prior_mean', '$' + m_prior_mean + '$', 1.0, \
					   pop_level))
pgm.add_node(daft.Node('m_prior_sig', '$' + m_prior_sig + '$', 2.0, \
					   pop_level))
pgm.add_node(daft.Node('class_prob', '$' + class_prob + '$', 3.0, \
					   pop_level))

# object priors
pgm.add_node(daft.Node('pr_m_true_i', \
					   '$' + pr(m_true + '|' + m_prior_mean_k + ',' + \
								m_prior_sig_k) + '$', \
					   1.5, obj_prior_level, aspect = 3.8, \
					   plot_params=s_color, shape='rectangle'))
pgm.add_node(daft.Node('pr_class_i', \
					   '$' + pr(class_true + r'|' + \
					   			class_probs) + '$', \
					   3.0, obj_prior_level, aspect = 1.4, \
					   plot_params=s_color, shape='rectangle'))

# object-level parameters
pgm.add_node(daft.Node('m_true_i', '$' + m_true + '$', \
					   2.0, obj_level))
pgm.add_node(daft.Node('class_i', '$' + class_true + '$', \
					   3.0, obj_level))

# likelihoods
pgm.add_node(daft.Node('pr_m_obs_ij', \
					   '$' + pr(m_obs + '|' + m_true + ',' + m_obs_sig) + '$', \
					   2.0, like_level, aspect=3.0, plot_params=s_color, \
					   shape='rectangle'))

# observables
pgm.add_node(daft.Node('m_obs_ij', '$' + m_obs + '$', 2.0, data_level, \
					   observed=True))
pgm.add_node(daft.Node('m_obs_sig_ij', '$' + m_obs_sig + '$', 1.0, data_level, \
					   fixed=True))

# edges
pgm.add_edge('prior_info', 'pr_m_prior_mean')
pgm.add_edge('prior_info', 'pr_m_prior_sig')
pgm.add_edge('prior_info', 'pr_class_prob')
pgm.add_edge('pr_m_prior_mean', 'm_prior_mean')
pgm.add_edge('pr_m_prior_sig', 'm_prior_sig')
pgm.add_edge('pr_class_prob', 'class_prob')
pgm.add_edge('m_prior_mean', 'pr_m_true_i')
pgm.add_edge('m_prior_sig', 'pr_m_true_i')
pgm.add_edge('class_prob', 'pr_class_i')
pgm.add_edge('pr_class_i', 'class_i')
pgm.add_edge('class_i', 'pr_m_true_i')
pgm.add_edge('pr_m_true_i', 'm_true_i')
pgm.add_edge('m_true_i', 'pr_m_obs_ij')
pgm.add_edge('pr_m_obs_ij', 'm_obs_ij')
pgm.add_edge('m_obs_sig_ij', 'm_obs_ij')

# object plate
pgm.add_plate(daft.Plate([0.5, data_level - 0.5, 3.0, \
						  pop_level - data_level - 0.02], \
						  label=r"$i < n_{\rm star}$", \
						  shift=-0.0, rect_params={"ec": "r"}, \
						  label_offset=(2, 2)))
pgm.add_plate(daft.Plate([0.78, data_level - 0.28, 2.02, 1.58], \
						  label=r"$j < n_{\rm obs}$", \
						  shift=-0.0, rect_params={"ec": "r"}, \
						  label_offset=(2, 2)))
pgm.add_plate(daft.Plate([0.5, pop_level - 0.48, 3.0, \
						  pop_prior_par_level - pop_level + 1], \
						  label=r"$k < n_{\rm cls}$", \
						  shift=-0.0, rect_params={"ec": "r"}, \
						  label_offset=(2, 158)))

# render and save
pgm.render()
pgm.figure.savefig('bhm_plot_1dg_class_obs_pdfs.pdf')

