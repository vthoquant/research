# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 15:23:02 2021

@author: vivin
"""
import numpy as np
import math
import statsmodels
import statsmodels.api as sm
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import time

class MARKET_DATA(object):
    def __init__(
        self,
        spot,
        rr,
        brw,
        cash_div,
        sigma
    ) :
        self.spot = spot
        self.rr = rr
        self.brw = brw
        self.sigma = sigma
        self.cash_div = cash_div
        
class BlackScholes_MonteCarlo(object):
    
    def __init__(
        self,
        mkt_data,
        paths,
        N,
        T,
        is_cash_div = False,
        seed = 5
    ) :
        self.mkt_data = mkt_data
        self.paths = paths
        self.is_cash_div = is_cash_div
        self.N = N
        self.steps = np.arange(0, N+1, 1)
        self.t_steps = [T*i/N for i in self.steps]
        self.fwd_unadj = [self.mkt_data.spot * math.exp(self.mkt_data.rr - self.mkt_data.brw) * x for x in self.t_steps]        
        np.random.seed(seed)
        norms = np.random.normal(0, 1, (paths, N))
        self.t_diffs = np.diff(self.t_steps, 1)
        scale_arr = [math.sqrt(x) for x in self.t_diffs]
        scale_mat = np.full((paths, N), scale_arr)
        self.norms_scaled = np.multiply(norms, scale_mat)
        self.spots = np.full((paths, N+1), self.mkt_data.spot)
        self.avg_spots = []
        
    def evaluate_paths(self):
        div_accum_array = []
        div_pv_array = []
        prop_div_array = []
        
        for step in self.steps:
            if step == 0:
                div_accum_array.append(0.0)
                div_pv_array.append(1.0)
                prop_div_array.append(0.0)
            else:
                prev_div_accum = div_accum_array[step - 1]
                new_div_accum = prev_div_accum * math.exp(self.mkt_data.rr * (self.t_steps[step] - self.t_steps[step - 1])) + self.mkt_data.cash_div
                div_accum_array.append(new_div_accum)
                div_pv_array.append(1 - new_div_accum/self.fwd_unadj[step])
                prop_div_array.append(1 - div_pv_array[step]/div_pv_array[step - 1])
                
                #diffuse spots
                prev_spots = self.spots[:, step - 1]
                norms_t = self.norms_scaled[:, step - 1]
                t_diff = self.t_diffs[step - 1]
                z_arr = np.exp(self.mkt_data.sigma * norms_t - np.square(self.mkt_data.sigma) * 0.5 * t_diff + (self.mkt_data.rr - self.mkt_data.brw) * t_diff)
                new_spots = np.multiply(prev_spots, z_arr)
                new_spots = new_spots - self.mkt_data.cash_div if self.is_cash_div else new_spots * (1.0 - prop_div_array[step])
                self.spots[:, step] = new_spots
                self.avg_spots.append(np.mean(new_spots))
    

class AMERICAN_OPTION(object):
    path = "C://Users//vivin//Documents//Temp//"
    
    def __init__(
        self, 
        K,
        T,        
        is_amer=True,
        is_call=True,
        mkt_data=None,
        model=None,
        plot = False
    ) :
        self.K = K
        self.T = T
        self.is_amer = is_amer
        self.is_call = is_call
        self.model = model
        self.mkt_data = mkt_data
        N = self.model.N
        self.intr = np.full((self.model.paths, N+1), 0.0)
        self.spv = np.full((self.model.paths, N+1), 0.0)
        self.epv = np.full((self.model.paths, N+1), 0.0)
        self.exercise_times = np.full((self.model.paths, ), self.model.steps[-1])
        self.plot = plot
        self.payoff_evaluated = False
        
    def evaluate_payoffs(self):
        self.intr = np.maximum(self.model.spots - self.K, 0.0) if self.is_call else np.maximum(self.K - self.model.spots, 0.0)
        for step in reversed(self.model.steps):
            curr_spots = self.model.spots[:, step]
            if step == self.model.steps[-1]:
                self.spv[:, step] = self.intr[:, step]
                self.epv[:, step] = self.intr[:, step]                
                #for plotting
                next_spv = self.spv[:, step]
            else:
                next_spv = self.spv[:, step + 1] * math.exp(-self.mkt_data.rr * self.model.t_diffs[step])
                ols = sm.GLM(
                    next_spv, 
                    sm.add_constant(
                        np.column_stack(
                            (
                                curr_spots,
                                np.square(curr_spots),
                                np.power(curr_spots, 3),
                                np.power(curr_spots, 4)
                            )
                        )
                    )
                )
                
                res_ols = ols.fit()
                self.epv[:, step] = res_ols.fittedvalues
                self.spv[:, step] = np.where(self.is_amer and np.logical_and(self.intr[:, step] > self.epv[:, step], self.epv[:, step] > 0), self.intr[:, step], next_spv)
                self.exercise_times = np.where(self.is_amer and np.logical_and(self.intr[:, step] > self.epv[:, step], self.epv[:, step] > 0), step, self.exercise_times)
                if self.plot:
                    self.plot_result(self.epv[:, step], curr_spots, next_spv, "dummy title", self.path)
                    
        self.payoff_evaluated = True
        
    def price(self, reevaluate=False):
        if not self.payoff_evaluated or reevaluate:
            self.evaluate_payoffs()
        return np.mean(self.spv[:, 0])
    
    def price_fwd(self):
        return math.exp(-self.mkt_data.rr * self.model.t_steps[-1]) * (self.model.avg_spots[-1] - self.K)
    
    @staticmethod
    def check_putcall_parity(pricer):
        pricer_call = AMERICAN_OPTION(pricer.K, pricer.T, False, True, pricer.mkt_data, pricer.model, False)
        pricer_put = AMERICAN_OPTION(pricer.K, pricer.T, False, False, pricer.mkt_data, pricer.model, False)
        pc_par = pricer_call.price() - pricer_put.price() - pricer_call.price_fwd()
        return pc_par
                    
    @staticmethod
    def plot_result(fit_y, samples_x, samples_y, title="", path="C:\\Temp"):
        plt.figure()
        plt.plot(samples_x, samples_y, "ro", label="samples")
        plt.plot(samples_x, fit_y, "bo", label="fit values")
        plt.title(title)
        plt.xlabel("spots")
        plt.ylabel("spv/epv")
        plt.legend()
        plt.savefig(path + title + ".png")
    
    def raw_data_generate(self):
        self.norm_df = pd.DataFrame(data=self.model.norms_scaled, columns=self.model.t_steps[1:])
        self.spots_df = pd.DataFrame(data=self.model.spots, columns=self.model.t_steps)
        self.intr_df = pd.DataFrame(data=self.intr, columns=self.model.t_steps)
        self.spv_df = pd.DataFrame(data=self.spv, columns=self.model.t_steps)
        self.epv_df = pd.DataFrame(data=self.epv, columns=self.model.t_steps)
        self.ex_times_df = pd.DataFrame(data=self.exercise_times, columns=[0])
        
    def raw_data_write(self):
        writer = pd.ExcelWriter(self.path + "data_raw.xlsx", engine="xlsxwriter")
        self.norm_df.to_excel(writer, sheet_name="normals", startrow=0, startcol=0)
        self.spots_df.to_excel(writer, sheet_name="spots", startrow=0, startcol=0)
        self.intr_df.to_excel(writer, sheet_name="intrinsic", startrow=0, startcol=0)
        self.spv_df.to_excel(writer, sheet_name="spv", startrow=0, startcol=0)
        self.epv_df.to_excel(writer, sheet_name="epv", startrow=0, startcol=0)
        self.ex_times_df.to_excel(writer, sheet_name="exer times", startrow=0, startcol=0)
        writer.save()
        
def main(
    spot=100.0,
    strike=100.0,
    tenor=1.0,
    rr=0.01,
    brw=0.001,
    cash_div=0.001,
    iv=0.2,
    is_call=True,
    is_amer=True,
    is_cash_div=True,
    mc_paths=20000,
    time_steps=10,
    plot=False
):
    mkt_data = MARKET_DATA(spot, rr, brw, cash_div, iv)
    model_bs_mc = BlackScholes_MonteCarlo(mkt_data, mc_paths, time_steps, tenor, is_cash_div)
    model_bs_mc.evaluate_paths()
    option_pricer = AMERICAN_OPTION(strike, tenor, is_amer, is_call, mkt_data, model_bs_mc, plot)
    price = option_pricer.price()
    print("price of instrument is {}".format(price))
    option_pricer.raw_data_generate()
    option_pricer.raw_data_write()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parse arguments')
    parser.add_argument("--spot", default=100.0, type=float)
    parser.add_argument("--strike", default=100.0, type=float)
    parser.add_argument("--tenor", default=1.0, type=float)
    parser.add_argument("--riskfreerate", default=0.1, type=float)
    parser.add_argument("--borrow", default=0.0, type=float)
    parser.add_argument("--cash_div", default=0.0, type=float)
    parser.add_argument("--impliedvol", default=0.00001, type=float)
    parser.add_argument("--is_call", default=True, type=bool)
    parser.add_argument("--is_amer", default=True, type=bool)
    parser.add_argument("--is_cash_div", default=True, type=bool)     
    parser.add_argument("--mc_paths", default=20000, type=int)
    parser.add_argument("--time_steps", default=10, type=int)
    parser.add_argument("--plot", default=False, type=bool)
    args = parser.parse_args()
    start_time = time.time()
    main(
        spot=args.spot,
        strike=args.strike,
        tenor=args.tenor,
        rr=args.riskfreerate,
        brw=args.borrow,
        cash_div=args.cash_div,
        iv=args.impliedvol,
        is_call=args.is_call,
        is_amer=args.is_amer,
        is_cash_div=args.is_cash_div,
        mc_paths=args.mc_paths,
        time_steps=args.time_steps,
        plot=args.plot
    )
    end_time = time.time()
    print("time taken: {} seconds".format(end_time - start_time))
    