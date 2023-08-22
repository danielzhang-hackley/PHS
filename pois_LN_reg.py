import numpy as np
import pandas as pd
import math
import torch
from torch.distributions import StudentT
import matplotlib.pyplot as plt
torch.set_default_dtype(torch.float64)

def normgampdf(mu, tau, a, b, m, t, logmode=False):
    # argument checks
    if not (mu.shape == tau.shape):
        raise Exception('Size of mu and tau must be the same size (mu is %s, tau is %s)' % (mu.shape, tau.shape))
    if not (a.shape == b.shape == m.shape == t.shape):
        raise Exception('Size of a/b/m/t parameters must be the same size (a: %s, b: %s, m: %s, t: %s)' %
                        (a.shape, b.shape, m.shape, t.shape))

    pix2 = torch.tensor(2. * math.pi)
    if not logmode:
        p = tau ** (a - 1) * (-tau / b).exp() * (-t * tau / 2 * (mu - m) ** 2).exp() * tau.sqrt()
        z = b ** a * pix2.sqrt() * a.lgamma().exp() / t.sqrt()
        p = p / z
    else:
        p = (a - 1) * tau.log() - tau / b - t * tau / 2 * (mu - m) ** 2 + tau.log() / 2
        z = a * b.log() + pix2.sqrt().log() + a.lgamma() - t.log() / 2
        p = p - z
    return p

def mvnqrat(th0, thP, muF, muR, hessF, hessR, hessFinv, hessRinv, ihsf):
    qrat = (-torch.log(torch.det(-hessRinv / ihsf)) + (th0 - muR).view(1, -1) @ hessR * ihsf @ (th0 - muR) -
            (-torch.log(torch.det(-hessFinv / ihsf)) + (thP - muF).transpose(0, 1) @ hessF * ihsf @ (thP - muF))) / 2
    return qrat

class MuTauSampler:
    def __init__(self, x, c, p):
        self.x = x
        self.c = c
        self.p = p
        self.tdistribution = StudentT(p['epsi_nu'])

    def epsisamp(self, epsi, beta0, tau, mu, epsi_2, tau_2):
        h, h_inv = 0, 0
        mu_f = epsi
        for i in range(0, 100):
            h = -torch.exp(mu + self.c * beta0 + torch.div(mu_f, tau.sqrt()) + self.p['y'][0] + epsi_2 / tau_2.sqrt()) / tau - 1
            h_inv = 1. / h
            # N - R update
            grad = self.x / tau.sqrt() - torch.exp(mu + self.c * beta0 + mu_f/ tau.sqrt() + self.p['y'][0] + epsi_2 / tau_2.sqrt())/tau.sqrt() - mu_f
            mu_f = mu_f - h_inv * grad

            # we've reached a local maximum
            if grad.norm() < 1e-6:
                break

        epsi_p = mu_f + h_inv.neg().sqrt() * self.tdistribution.sample(torch.Size([self.p['length'], 1]))

        pr = epsi_p * self.x / tau.sqrt() - torch.exp(mu + self.c * beta0+ epsi_p / tau.sqrt() + self.p['y'][0] +
                epsi_2 / tau_2.sqrt()) - epsi_p ** 2 / 2 - (epsi * self.x / tau.sqrt() - torch.exp(mu + self.c *
                    beta0 + epsi / tau.sqrt() + self.p['y'][0] + epsi_2 / tau_2.sqrt()) - epsi ** 2 / 2)

        arat = pr + self.tqrat(epsi, epsi_p, mu_f, mu_f, h_inv.neg().sqrt(), h_inv.neg().sqrt())

        ridx = torch.rand(self.p['length'], 1).log() >= arat.clamp(max=0)
        ridx_float = ridx.type(torch.float64)  # need to check
        epsi[~ridx] = epsi_p[~ridx]
        mrej = (1 - ridx_float).mean()
        return epsi, mrej

    def tqrat(self, th_0, th_p, mu_f, mu_r, sig_f, sig_r):
        qrat = -torch.log(sig_r) - (self.p['epsi_nu'] + 1) / 2 * torch.log(1 + (th_0 - mu_r) ** 2 /
                (self.p['epsi_nu']* sig_r ** 2)) - (-torch.log(sig_f) - (self.p['epsi_nu'] + 1) / 2 * torch.log(1 +
                (th_p - mu_f) ** 2 / (self.p['epsi_nu'] * sig_f ** 2)))
        return qrat

    def betasamp(self, Beta, epsi, mu, tau, epsi_2):
        mu = torch.full((self.p['reidx'].size()[0], 1), mu[0])
        tau_2 = torch.full((self.p['reidx'].size()[0], 1), tau[0])
        tau = tau[tau.size()[0] - 1]
        muF, HF, HFinv, is_max = self.betaNR(Beta, epsi, mu, tau, epsi_2, tau_2)
        BetaP = self.mvtrnd_jh(muF, -HFinv, 1, 0)
        if not is_max:
            muR, HR, HRinv, is_MAX = self.betaNR(BetaP, epsi, mu, tau, epsi_2, tau_2)
        else:
            muR = muF
            HR = HF
            HRinv = HFinv

        # TODO m = len(muF)
        m = 1
        pr = self.p['cx'] * BetaP - torch.exp(mu + self.c * BetaP + epsi / tau.sqrt() + self.p['y'][0] + torch.div(epsi_2,
                    tau_2.sqrt())).sum() + self.logmvnpdf(BetaP) - (self.p['cx'] * Beta - torch.exp(mu + self.c * Beta +
                    epsi / tau.sqrt() + self.p['y'][0] + torch.div(epsi_2, tau_2.sqrt())).sum() + self.logmvnpdf(Beta))
        qrat = (-(torch.det(-HRinv)).log() - (self.p['beta_nu'] + m) * (1 - (Beta - muR).view(1, -1) * HR * (Beta - muR)
                / self.p['beta_nu']).log() - (-(torch.det(-HFinv)).log() - (self.p['beta_nu'] + m) *
                (1 - (BetaP - muF).view(1, -1) * HF * (BetaP - muF)/self.p['beta_nu']).log()))/2
        arat = pr + qrat
        if torch.rand(1).log() < arat.clamp(max=0):
            Beta = BetaP
            rej = 0
        else:
            rej = 1
        return Beta, rej

    def betaNR(self, Beta, epsi, mu, tau, epsi_2, tau_2):
        is_max = False
        HF, HFinv = 0, 0

        for i in range(0, self.p['tMH_NR_iter']):

            gr = self.p['cx'] - torch.exp(mu + self.c * Beta + torch.div(epsi, tau.sqrt()) + self.p['y'][0] +
                epsi_2 / tau_2.sqrt()).view(1, -1) @ self.c - (self.p['invcovmu'] * (Beta - self.p['mubeta'])).view(1, -1).view(1,-1)
            HF = -torch.reshape(self.p['cc'] @ torch.exp(mu + self.c * Beta + epsi / tau.sqrt() + self.p['y'][0] +
                                epsi_2 / tau_2.sqrt()), (self.c.size()[1], self.c.size()[1])) - self.p['invcovmu']
            HFinv = torch.inverse(HF)
            # check if we've reached a local maximum (Hessian should be negative definite)

            temp = gr.norm() <= 1e-6
            if temp and (HF.eig()[0][:, 0] < 0).all().item():
                is_max = True
                break
            # N - R update
            Beta = Beta - HFinv * gr  # below coded in matlab
        return Beta, HF, HFinv, is_max

    def mvtrnd_jh(self, mu, sigma, cases=None, skip=None):
        if cases is None:
            cases = 1
        if skip is None:
            skip = 0

        if ~skip:
            # sanity checks
            if sigma.dim() != 2:
                print('Covariance matrix must be a 2D!')
            sz = sigma.size()
            if sz[0] != sz[1]:
                print('Covariance matrix must be square!')
            # TODO check 'Mean must be a column vector with matching dimensions!' from mvtrnd_jh.m
            # TODO check 'Covariance matrix must be symmetric and positive-semidefinite!'
        m = sigma.size()[0]
        V = sigma.size()[1]
        l, V = torch.eig(sigma, eigenvectors=True)
        V = V[0, 0]
        l = l[0, 0]
        x = mu + V * l.sqrt() * self.tdistribution.sample(torch.Size([m, cases]))
        return x

    def logmvnpdf(self, x): # assumes icmat is already inverted to save compute
        p = -(x - self.p['mubeta']).view(1, -1) @ self.p['invcovmu'] @ (x - self.p['mubeta'])/2
        return p

    def mutausamp(self, mu, tau, Beta, epsi, tau_2, epsi_2, j):
        # select jth mu / tau from array
        mu = mu[j]
        tau = tau[j]

        # TODO: figure out sparse tensor
        tidx = self.p['reidx_bin'][:, j]
        tidx[tidx != 0] = 1
        tidx = tidx.type(torch.bool)
        epsi = epsi[tidx]
        epsi_2 = epsi_2[tidx]

        # constant variables
        if len(self.p['y']) > 1:
            y = self.p['y'][tidx]
        else:
            y = self.p['y'][0]
        Y = pd.Series([j, epsi.view(1, -1) @ self.x[tidx], torch.exp(self.c[tidx, :] @ Beta + y),
                       torch.exp(epsi_2 / tau_2.sqrt())], index=['j', 'eps_x', 'eBC', 'exp_eps_2_tau_2'])
        Y['eBC'] = torch.reshape(Y['eBC'], (len(Y['eBC']), 1))
        mutau = torch.tensor([[mu], [torch.log(tau)]])
        muF, HF, HFinv, is_max, use_LS = self.mutauNR(mu, torch.log(tau), epsi, Y)

        # now propose with multivariate Gaussian centered at MAP (tau log transformed) with covariance matrix from Hessian
        mean = torch.reshape(muF, (1, 2)).squeeze(0)
        cov = (-HFinv * self.p['mutau_ihsf']).cpu().detach().numpy()
        mutauP = np.random.multivariate_normal(mean, cov)
        mutauP = torch.reshape(torch.from_numpy(mutauP), (2, 1))
        mutauP = mutauP.type(torch.float64)

        if not is_max:
            muR, HR, HRinv, is_MAX, USE_LS = self.mutauNR(muF[0], muF[1], epsi, Y)
        else:
            muR = muF
            HR = HF
            HRinv = HFinv
        arat = self.pratmutau(mutau[0], mutau[1], mutauP[0], mutauP[1], epsi, Y) + \
               mvnqrat(mutau, mutauP, muF, muR, HF, HR, HFinv, HRinv, self.p['mutau_ihsf'])

        if torch.rand(1).log() < arat.clamp(max=0):
            mu = mutauP[0]
            tau = torch.exp(mutauP[1])
            rej = 0
        else:
            rej = 1
        return mu, tau, rej, use_LS

    def mutauNR(self, mu, tau, epsi, Y):
        is_max = False  # if we converged to a maximum
        is_NR_bad = False  # if regular N-R iterations are insufficent, so we need to employ line search
        use_LS = False  # whether we used linesearch (for display purposes only)
        mu_prev = mu
        tau_prev = tau

        Hinv = torch.ones(2, 2) * torch.tensor(float('nan'))
        i = 1
        while True:
            # print('mutauNR times %s' % i)
            tau_e = torch.exp(tau)
            if i == 1:
                Y = Y.append(pd.Series([torch.exp(epsi / tau_e.sqrt())], index=['exp_eps_tau']))
                Y = Y.append(pd.Series([Y['eBC'] * Y['exp_eps_tau'] * Y['exp_eps_2_tau_2']], index=['d1']))
            else:
                Y['exp_eps_tau'] = torch.exp(epsi / tau_e.sqrt())
                Y['d1'] = Y['eBC'] * Y['exp_eps_tau'] * Y['exp_eps_2_tau_2']

            grad = torch.tensor([self.p['sums'][Y['j']] - torch.exp(mu) * sum(Y['d1']) - self.p['mutau'][Y['j']]
                                 * tau_e * (mu - self.p['mumu'][Y['j']]), (-Y['eps_x'] + torch.exp(mu) *
                                (Y['d1'].view(1, -1) @ epsi)) / (2*tau_e**(3/2)) + (2 * self.p['taua'][Y['j']] - 1) /
                                 (2 * tau_e) - 1 / self.p['taub'][Y['j']] - self.p['mutau'][Y['j']] / 2 *
                                 (mu - self.p['mumu'][Y['j']]) ** 2])
            grad = torch.reshape(torch.Tensor(grad), (2, 1))
            # H = hessmutau(mu, tau_e, Beta, epsi, X, Y)
            H = torch.ones(2, 2) * torch.tensor(float('nan'))
            H[0, 0] = -torch.exp(mu) * sum(Y['d1']) - self.p['mutau'][Y['j']] * tau_e
            H[1, 1] = 3 / 4 * tau_e ** (-5 / 2) * Y['eps_x'] - torch.exp(mu) * \
                      ((Y['d1'].view(1, -1) @ epsi) * (3/4) * tau_e ** (-5/2) +
                       (Y['d1'].view(1, -1)@(epsi ** 2))*(tau_e ** -3)/4) - \
                      (2 * self.p['taua'][Y['j']] - 1) / (2 * tau_e ** 2)
            H[0, 1] = torch.exp(mu) * (Y['d1'].view(1, -1) @ epsi) / (2 * tau_e ** (3/2)) - \
                      self.p['mutau'][Y['j']] * (mu - self.p['mumu'][Y['j']])
            H[1, 0] = H[0, 1]

            # change-of-variable chain rule factors
            H[1, 1] = grad[1] * tau_e + tau_e ** 2 * H[1, 1]
            H[0, 1] = tau_e * H[0, 1]
            H[1, 0] = H[0, 1]
            grad[1] = grad[1] * tau_e
            # change - of - variable Jacobian factors
            grad[1] = grad[1] + 1

            # if Hessian is problematic, rewind an iteration and use line search by default
            h_np = H.numpy()
            eps = 2.2204e-16  # pytorch doesn't have eps

            if (1 / np.linalg.cond(h_np, 1) < eps) or H[:].isnan().any().item() or H[:].isinf().any().item():
                # if we're in a problematic region at the first iteration, break and hope for the best
                if i == 1:
                    H = torch.eye(2)
                    Hinv = -torch.eye(2)
                    print("WARNING: Full conditional shifted significantly since last iteration!")
                    break
                is_NR_bad = True
                i -= 1
                mu = mu_prev
                tau = tau_prev
                continue
            else:
                is_NR_bad = False
            # check if Hessian is negative definite
            is_ndef = (not is_NR_bad) and (H.eig()[0][:, 0] < 0).all().item()

            # if Newton-Raphson will work fine, use it
            Hinv = H.inverse()
            step = -Hinv @ grad

            if (grad.norm() <= 1e-6) and is_ndef:
                is_max = True
                break

            # 2. otherwise, employ line search
            fc_step = self.fcmutau(mu + step[0], tau + step[1], epsi, Y)
            if is_NR_bad or fc_step.isnan().item() or fc_step.isinf().item() \
                    or (fc_step - self.fcmutau(mu, tau, epsi, Y) < -1e-3):
                print('line search ...')
                # indicate that we used linesearch for these iterations
                use_LS = True

                # 2.1. ensure N-R direction is even ascending. if not, use direction of gradient
                if not is_ndef:
                    step = grad

                # 2.2. regardless of the method, perform line search along direction of step
                s0 = step.norm()
                d_hat = step / s0
                # bound line search from below at current value
                fc = self.fcmutau(mu, tau, epsi, Y) * torch.tensor([1, 1])

                # 2.3. do line search
                for l in range(0, 50):
                    s = s0 * 0.5 ** (l - 1)
                    f = self.fcmutau(mu + s * d_hat[0], tau + s * d_hat[1], epsi, Y)
                    # need to fix indexing?
                    if (fc[0][(l - 1) % 2] > fc[0][(l - 2) % 2]) and (fc[0][(l - 1) % 2] > f):
                        # correct indexing for python?
                        s = s0 * 0.5 ** (l - 1)
                        break
                    else:
                        fc[0][l % 2] = f

                step = d_hat * s
            # update mu, tau
            mu_prev = mu
            tau_prev = tau
            mu = mu + step[0]
            tau = tau + step[1]

            i += 1
            if i > self.p['tMH_NR_iter']:
                if not is_ndef:
                    H = torch.eye(2)
                    Hinv = -torch.eye(2)
                    print('WARNING: Newton-Raphson terminated at non-concave point!')
                break

        mutau = torch.tensor([[mu], [tau]])
        return mutau, H, Hinv, is_max, use_LS

    def fcmutau(self, mu, tau, epsi, Y):
        tau_e = torch.exp(tau)
        fc = mu.type(torch.float64) * self.p['sums'][Y['j']] + Y['eps_x'] / tau_e.sqrt() - mu.exp().type(torch.float64) * \
             ((Y['eBC'] * Y['exp_eps_2_tau_2'].type(torch.float64)).view(1, -1) @ (epsi / tau_e.sqrt()).exp().type(torch.float64)) + \
             normgampdf(mu, tau_e, self.p['taua'][Y['j']], self.p['taub'][Y['j']], self.p['mumu'][Y['j']],
                        self.p['mutau'][Y['j']], True) + tau
        fc = fc.type(torch.float64)
        return fc

    def pratmutau(self, mu, tau, muP, tauP, epsi, Y):

        pr = self.fcmutau(muP, tauP, epsi, Y) - self.fcmutau(mu, tau, epsi, Y)
        return pr

########################

x = torch.poisson(torch.exp(-3. + 0.9 * torch.randn(50000, 1)))
c = torch.zeros_like(x)
x_inp = x.size()
p = pd.Series([2000, 50, torch.tensor([[-3.]]), torch.tensor([[10.]]), torch.tensor([[0.2]])],
                  index=['niter', 'burnin', 'mumu', 'taua', 'taub'])
p = p.append(pd.Series(x_inp[0], index=['length']))
p = p.append(pd.Series([[0.]], index=['y']))
p = p.append(pd.Series([torch.ones_like(x)], index=['reidx']))
p = p.append(pd.Series(p['reidx'].max(), index=['ncat']))
# TODO: X.sums = accumarray(X.reidx, X.x)
p = p.append(pd.Series([[x.sum()]], index=['sums']))
# TODO: X.cx = X.x'*X.c,X.c(isnan(X.c)) = 0,X.c(isinf(X.c)) = 0
p = p.append(pd.Series([x.view(1, -1) @ c], index=['cx']))

# binary expansion of self.reidx for quick slicing
# self.reidx_bin = torch.sparse_coo_tensor(torch.stack((torch.arange(0, self.length, step=1),
#                   torch.ones(self.length))), torch.ones(self.length))
p = p.append(pd.Series([torch.ones(p['length'], p['reidx'].size()[1])], index=['reidx_bin']))

# necessary for computing Hessian of p(beta | -)
cc = torch.ones(x_inp[1], x_inp[1], p['length']) * torch.tensor(float('nan'))
for i in range(0, p['length']):
    cc[:, :, i] = c[i, :].reshape(-1, 1) * c[i, :]
cc = cc.reshape(x_inp[1] ** 2, p['length'])  # self.cc is a matrix
p = p.append(pd.Series([cc], index=['cc']))
del cc

# M-H inverse Hessian scaling factor (to tweak M-H proposal dist.)
p = p.append(pd.Series(100, index=['tMH_NR_iter']))
p = p.append(pd.Series(20, index=['beta_nu']))
p = p.append(pd.Series(1, index=['mutau_ihsf']))
p = p.append(pd.Series(1, index=['tau_ihsf']))
p = p.append(pd.Series(1, index=['epsi_ihsf']))
p = p.append(pd.Series(5, index=['epsi_nu']))

# Hyperparameters
# p = p.append(pd.Series(torch.zeros(p['ncat'], 1), index=['mumu']))
p = p.append(pd.Series([torch.ones(p['ncat'].type(torch.int), 1)], index=['mutau']))
# p = p.append(pd.Series(torch.ones(p['ncat'] + 1, 1), index=['taua']))
# p = p.append(pd.Series(10 * torch.ones(p['ncat'] + 1, 1), index=['taub']))

# beta ~ normal_m(mubeta, covmu)
p = p.append(pd.Series([2 * torch.eye(x_inp[1])], index=['covmu']))
p = p.append(pd.Series([torch.inverse(p['covmu'])], index=['invcovmu']))
p = p.append(pd.Series([torch.zeros(x_inp[1], 1)], index=['mubeta']))

# Initialization tau / mu
t0 = torch.ones(p['ncat'].type(torch.int) + 1, 1)
m0 = torch.log(torch.mean(x))  # user specified
tau = torch.ones(p['ncat'].type(torch.int) + 1, p['niter']) * torch.tensor(float('nan'))
tau[:, 0] = t0[:,0]
# tau_2 = tau[len(tau[:, 0])-1, 0]
mu = torch.ones(p['ncat'].type(torch.int), p['niter']) * torch.tensor(float('nan'))
mu[:, 0] = m0

# epsilon
epsi = torch.zeros_like(x)
epsi_ch_ar = torch.tensor(float('nan'))

# beta
p = p.append(pd.Series([torch.zeros(x_inp[1])], index=['beta0']))
fixed_beta = torch.empty(0)

# whether to use hierarchical model.  otherwise, categories are treated independently
ignore_categories = False

##############################################################
X = MuTauSampler(x, c, p)
del x, c
epsi_h, epsi_h_ar = X.epsisamp(torch.zeros(x_inp[0], 1), p['beta0'], torch.full((p['reidx'].size()[0], 1), tau[0, 0]),
                               torch.full((p['reidx'].size()[0], 1), mu[0, 0]), epsi, tau[tau.size()[0] - 1, 0])

# smooth out initial epsilon samples
for i in range(0, 5):
    epsi_h, epsi_h_ar = X.epsisamp(epsi_h, p['beta0'], torch.full((p['reidx'].size()[0], 1), tau[0, 0]),
                                   torch.full((p['reidx'].size()[0], 1), mu[0, 0]), epsi, tau[tau.size()[0] - 1, 0])
# beta
Beta = torch.ones(x_inp[1], p['niter']) * torch.tensor(float('nan'))

if len(fixed_beta) == 0:
    if len(p['beta0']) == 0:
        print('Initializing beta chain')
        Beta[:, 0] = X.betasamp(torch.zeros(x_inp[1], 1), epsi, m0, t0)
    else:
        print('Using user-supplied beta0')
        if len(p['beta0']) != x_inp[1]:
            print('Length of P.beta0 must match number of covariates!')
        Beta[:, 0] = p['beta0']
else:
    if len(fixed_beta) != len(x_inp[1]):
        print('Fixed beta size mismatch!')
    Beta = X.repmat(fixed_beta, 1, p['niter'])

# whether Metropolis sample was rejected for beta and mu_j,tau_j, tau_0
mrej_b = torch.ones(p['niter'], 1) * torch.tensor(float('nan'))
mrej_mt = torch.ones(p['niter'], p['ncat'].type(torch.int)) * torch.tensor(float('nan'))
mrej_t0 = torch.ones(p['niter'], 1) * torch.tensor(float('nan'))

# full log likelihoods (for temperature diagnostics)
full_lik = torch.ones(p['niter'], 1) * torch.tensor(float('nan'))
# Primary MCMC iterations
burnin_flag = 0
last_full_lik_idx = 1

print('Running full posterior MCMC for %s iterations (burnin %s)' % (p['niter'], p['burnin']))

for i in range(1, p['niter']):
    # Metropolis / Gibbs draws
    # draw epsilons via M-H
    epsi_h, epsi_h_ar = X.epsisamp(epsi_h, Beta[:, i - 1], torch.full((p['reidx'].size()[0], 1), tau[0, i - 1]),
                                   torch.full((p['reidx'].size()[0], 1), mu[0, i - 1]), epsi,
                                   tau[tau.size()[0] - 1, i - 1])

    # draw beta via M-H
    if len(fixed_beta) == 0:
        Beta[:, i], mrej_b[i] = X.betasamp(Beta[:, i - 1], epsi, mu[:, i - 1], tau[:, i - 1], epsi_h)
    #  draw (mu_j, tau_j) via Gibbs sampling of full conditional
    mt_use_LS = False
    for j in range(0, p['ncat'].type(torch.int)):
        mu[j, i], tau[j, i], mrej_mt[i, j], use_LS = X.mutausamp(mu[:, i - 1], tau[:, i - 1], Beta[:, i], epsi_h,
                                                                 tau[tau.size()[0] - 1, i - 1], epsi, j)
        mt_use_LS = mt_use_LS | use_LS  # need to check if this is correct

    # draw tau_0 via Gibbs sampling of full conditional
    # skip the below code
    #   if ~P.ignore_categories && X.ncat > 1,
    #     [tau(end, i) mrej_t0(i)] = tausamp(tau(end, i - 1), mu(:, i), Beta(:, i), epsi, tau(:, i), epsi_h, X);
    #  else
    tau[tau.size()[0] - 1, i] = float('inf')

    if 1 - burnin_flag & i > p['burnin']:
        burnin_flag = 1

    if p['ncat'] <= 4:
        m_string = mu[:, i].cpu().detach().numpy()
        t_string = np.array2string(tau[:, i].cpu().detach().numpy(), precision=2, separator=', ')
        t_string = t_string[1:-1]
        temp = mrej_mt[max(i - 50, 1):i + 1, 0].cpu().detach().numpy()
        mrej_t_string = 1 - np.mean(temp)
    else:
        m_string = mu[:, i].cpu().detach().numpy()
        m_string = np.mean(m_string)
        t_string = tau[:, i].cpu().detach().numpy()
        t_string = 1 / np.mean(1. / t_string)
        t_string = t_string[1:-1]
        temp = mrej_mt[max(i - 50, 1):i + 1, 0].cpu().detach().numpy()
        mrej_t_string = 1 - np.mean(np.mean(temp))
    if mt_use_LS:
        warnstr = '[!]'
    else:
        warnstr = ''

    epsi_ch_ar_string = epsi_ch_ar.cpu().detach().numpy()
    temp = mrej_b[max(i - 50, 1):i + 1, 0].cpu().detach().numpy()
    mrej_b_string = 1 - np.mean(temp)
    full_like_string = full_lik[last_full_lik_idx].cpu().detach().numpy()

    print('[%s/%s] m = %.2f, t = %s, e. A.R. = %s, m/t A.R. = %.2f, b. A.R. = %0.2f, LL = %0.2f %s'
          % (i, p['niter'], m_string, t_string, epsi_ch_ar_string, mrej_t_string, mrej_b_string, full_like_string, warnstr))

# plot the result
u = mu[0, 50:2000].numpy()
sig = 1/tau[0, 50:2000].sqrt().numpy()
fig = plt.figure()
plt.plot(u, sig, 'o', color='blue', markersize=1)
plt.plot(-3, 0.9, '+', color='red', markersize=10, markeredgewidth=2)
fig.suptitle('MCMC draws from LNP posterior p(mu, sigma|x)', fontsize=16)
plt.xlabel('mu', fontsize=12)
plt.ylabel('sigma', fontsize=12)
fig.savefig('python_result.png')


