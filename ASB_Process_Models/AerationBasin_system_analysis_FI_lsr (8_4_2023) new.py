#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: 
    Fatima Iqbal fatimaiqbal0313@gmail.com 
    Christopher (Sutton) Page 
    Lewis Stetson Rowles


systems included:
    influent
    sedeimentation Tank
    Screw Press
    Aerated Stabilization Basin
    Effluent 
    
    
references:
    
Bryant, C. Updating a model of pulp and paper wastewater treatment in a partial-mix aerated stabilization basin system, 
Water Sci Technol (2010) 62 (6): 1248–1255.
https://doi.org/10.2166/wst.2010.934

Bryant, C. A simple method for analysis of the performance of aerated wastewater lagoons,
Water Sci Technol (1995) 31 (12): 211–218.
https://doi.org/10.2166/wst.1995.0489

Brazil, B.L.; Summerfelt, S.T. Aerobic treatment of gravity thickening tank supernatant,
Aquacultural Engineering (2006) 34 (2): 92-102
https://doi.org/10.1016/j.aquaeng.2005.06.001

Metcalf & Eddy, Wastewater Engineering: Treatment and Resource Recovery,
McGraw-Hill 2013, 5th edition

Juneidi, S.J., Sorour, M.T. and Aly, S.A., 2022. Proposed systematic approach for
assessing different wastewater treatment plants alternatives: Case study of Aqaba city (South Jordan).
Alexandria Engineering Journal, 61(12), pp.12567-12580.
https://doi.org/10.1016/j.aej.2022.06.044

https://hachcompany.custhelp.com/app/answers/answer_view/a_id/1020001/~/how-can-i-convert-between-nh4-%28ammonium%29-to-nh3-%28ammonia%29%3F-
    

The following wastewater (WW) constituents are used as inputs into the treatment system:
IN_COD     #mg/L
IN_SBOD    #mg/L
IN_NH3     #mg/L
IN_PO4     #mg/L
IN_TSS     #mg/L

Flowrate into the system is set: 
IN_Q       #m3/d



list of parameters that users can change in the Dash

IN_COD     #mg/L
IN_SBOD    #mg/L
IN_NH3     #mg/L
IN_PO4     #mg/L
IN_Q       #m3/d
Power      #W
solids_percent  #%

"""

from scipy.interpolate import interp1d
from scipy.integrate import odeint
import numpy as np
import pandas as pd
import lhs
import math
import scipy
from scipy import stats
from setup import setup_data
import copy


# %% LCC/LCA modeling inputs

# outputs of interest
output_perc_mid = 50
output_perc_low = 5
output_perc_high = 95


# general parameters - import spreadsheet tabs as dataframes
general_assumptions = pd.read_excel(
    'assumptions_AerationBasin.xlsx', sheet_name='General', index_col='Parameter')
design_assumptions = pd.read_excel(
    'assumptions_AerationBasin.xlsx', sheet_name='Design', index_col='Parameter')
cost_assumptions = pd.read_excel(
    'assumptions_AerationBasin.xlsx', sheet_name='Cost', index_col='Parameter')
LCA_assumptions = pd.read_excel(
    'assumptions_AerationBasin.xlsx', sheet_name='LCA', index_col='Parameter')


# number of Monte Carlo runs
n_samples = int(general_assumptions.loc['n_samples', 'expected'])

# create empty datasets to eventually store data for sensitivity analysis (Spearman's coefficients)
correlation_distributions = np.full((n_samples, n_samples), np.nan)
correlation_parameters = np.full((n_samples, 1), np.nan)
correlation_parameters = correlation_parameters.tolist()


# loading in all of the data
result = setup_data([general_assumptions, design_assumptions, LCA_assumptions],
                    correlation_distributions, correlation_parameters, n_samples)

# creates variables for each of the variables in the excel file

for key in result.keys():
    exec(f'{key} = result[key]')









# %% Aerated Stabilization Basin

# process models for ASB


# Flow in Aerated Stablization Basin is set to the Flow volume from the sedimentation Tank


Q = IN_Q * 3785.4118

Influent_BOD1 = Influent_BOD 

Benchmark_efficiency = 10

VSS = TSSi * 0.80

IN_PO4 = IN_PO4_z + Sup_PO4

IN_NH3 = IN_NH3_z + Sup_NH3

Area_ft2_1A = Area_1A * 10.7639

Area_ft2_1B = Area_1B * 10.7639

# STEP 1: Temperature of Pond1A and Pond1B

Ta = (Area_ft2_1A * 0.000012 * air_temp + IN_Q * Ti ) / ((Area_ft2_1A * 0.000012) + 1)

T_hello = ((Ta - 32) * 5) / 9

#Temperature of Pond1B

Tai = (Area_ft2_1B * 0.000012 * air_temp_1b + IN_Q * Ti_1B ) / ((Area_ft2_1B * 0.000012) + 1)

T_1B_hello = ((Tai - 32) * 5) / 9

# STEP 2: TEMP SENSITIVITY- Calculate Arrhenius temperature sensitivity coefficient

import numpy as np

def get_theta(temp):
    return np.where((temp >= 5) & (temp <= 15), 1.109, 
                    np.where((temp > 15) & (temp <= 30), 1.042, 0.967))

# Assuming T and T_1B are already defined NumPy arrays
theta_T = get_theta(T)
theta_T1_B = get_theta(T_1B)

print(f"Theta for System 1 (T): {theta_T}")
print(f"Theta for System 2 (T_1B): {theta_T1_B}")


# STEP 3: k.VSS- Adjust oxidation rate to cell temperature

K = 2.5 * (IN_PO4 / (0.053 + IN_PO4)) * (theta_T ** (T - 20))

#STEP 4: Supplemental P and N

IN_PO4 = IN_PO4_z + Sup_PO4

IN_NH3 = IN_NH3_z + Sup_NH3


# STEP 5: Calculation of Power


P_x = aeration / Volume  # W/m3

Power = P_x * Volume  # W

# STEP 6: MIXING- Calculate Mixing Intenstiy

Mixing_Intensity = Power / Volume  # W/m3


# STEP 7: PARTIAL MIX Calculate ration of cell mixing intensity to complete mix

Complete_Mixing = CM_Power / CM_Volume  # W/m3

Partial_Mix = Mixing_Intensity / Complete_Mixing  # fraction



# STEP 8: Cells in series- Select number of complete-mix cells to represent hydraulics and calculate first-order rate equation

n = 1

#Rate = {1 + K_t * tn} ** n
Rate = (1 + K * tn) ** n



# STEP 9 : Calculate the BOD5 removed

K = 2.5 * (IN_PO4 / (0.053 + IN_PO4)) * (theta_T ** (T - 20))

Effluent_BOD5 = 1 / (1 + (K * Hydrau_Reten)) * Influent_BOD1

BOD5_removal = Influent_BOD1 - Effluent_BOD5  # mg/L

x_numerator = Y * (Influent_BOD1 - Effluent_BOD5)

x_denomenator = 1 + (kd * Hydrau_Reten)

x = x_numerator / x_denomenator  # mg/L

Px = (x * Q) / 1000  # kg/d

Px_O2 = 1.42 * Px 

O2_requirement = Q * (BOD5_removal / (f * 1000))-(Px_O2) # kg/d

oxygen_supply1 = aeration * Coeff_Power

BOD_Effluent = ((oxygen_supply1 + Px_O2 ) / (Q * 1000)) 

BOD5_Effluent = Influent_BOD - BOD_Effluent

Power_req_bod = O2_requirement / (Coeff_Power)  # kW

P_x = aeration / Volume  # W/m3

Power = P_x * Volume

Supply_O = ((aeration * 24) / 1000) * Coeff_Power #kg O2 / d 

Eff_BOD_kg = (BOD5_removal / 1000000) * (Q * 1000)

O2_required_per_BOD = Eff_BOD_kg / O2_requirement

# STEP 10: Calculation of oxygen transfer rate under standard conditions

pressure = -(g * M * (elevation_a - 0)) / (R * (273.15 + T))

relative_pressure = math_exp ** pressure

C_20 = 1 + d_e * (Diffuser_depth / Pa)

Oxygen_Con_20 = C20 * C_20

O2_requirement_h = Supply_O / 24

SOTR1 = O2_requirement_h / (0.50 * Foul_factor)

SOTR2 = (beta * (C20 / C12) * relative_pressure * Oxygen_Con_20) - 2.0

SOTR3 = Oxygen_Con_20 / SOTR2

SOTR4 = SOTR3 * (1.024) ** (20 - T)

SOTR5 = SOTR1 * SOTR4                       #kg O2/ h

SOTR6 = SOTR5 * 26.4                        #kg O2/ d

EFF_BOD_O2 = SOTR6 / O2_required_per_BOD

EFF_BOD_O2_isr = (EFF_BOD_O2 * 1000000) / (Q * 1000)

# STEP 11: Calculation of oxygen requirement for BOD removal

O2_requirement_h_isr = O2_requirement / 24

SOTR1_isr = O2_requirement_h_isr / (0.50 * Foul_factor)

SOTR2_isr = (beta * (C20 / C12) * relative_pressure * Oxygen_Con_20) - 2.0

SOTR3_isr = Oxygen_Con_20 / SOTR2_isr

SOTR4_isr = SOTR3_isr * (1.024) ** (20 - T)

SOTR5_isr = SOTR1_isr * SOTR4_isr                       #kg O2/ h

SOTR6_isr = SOTR5_isr * 26.4                        #kg O2/ d

Effluent_BOD_trials1 = (K * Hydrau_Reten * (SOTR6 / SOTR6_isr))

K_isr = (IN_PO4 / (0.053 + IN_PO4)) * (theta_T ** (T - 20))

Effluent_BOD_trials2 = 1 + (K_isr * Hydrau_Reten) * (SOTR6  / SOTR6_isr ) 

Trial_new = Influent_BOD / Effluent_BOD_trials2

aerobic_biomass_yield = 0.5

bio_growth = aerobic_biomass_yield * (Influent_BOD1 - Effluent_BOD5)

 

# STEP 12: LBOD to SBOD-Conversion of SBOD6-120 to SBOD5

Exponent_cal = (math_exp) ** -kL * t_n

LBOD_to_SBOD = Initial_Concentration * Exponent_cal  # mg/L


# STEP 13: Aerobic growth-Calculate new aerobic biomass growth

Aerobic_growth = Influent_BOD1 - Trial_new

Aerobic_Biomass_growth = aerobic_biomass_yield * Aerobic_growth  # mg/L



# Step 28: Uptake N-Calculate nitrogen uptake by biomass growth

N_Uptake_ratio = 14

Overall_Biomass_growth = Aerobic_Biomass_growth

Uptake_N = N_Uptake_ratio / 115 * Overall_Biomass_growth  # mg/L

# Step 29: Uptake P-Calculate phosphorus uptake by biomass growth

P_Uptake_ratio = 2.2

Uptake_P = P_Uptake_ratio / 115 * Overall_Biomass_growth  # mg/L


# STEP 14: Temperature correction maximum specific growth

spec_growth_AOB = max_growth_coefficient * (1.072 ** (T - 20))

# STEP 15: Temperature correction endogenous decay coefficient

spec_endo_decay = b_20 * (1.029 ** (T - 20))

#STEP 16: Specific growth rate of ammonia oxidizing bacteria

growth_NH4 = spec_growth_AOB * (S_NH3 / (S_NH3 + K_NH4))

spec_growth_NH4 = growth_NH4 * (S_DO / (S_DO + K_AOB)) - spec_endo_decay

#STEP 17: Solid retention rate calculation

SRT = ( 1 / spec_growth_NH4) * 1.5

# STEP 18: Effluent Ammonia calculation

EFF_NH = spec_growth_AOB * (S_DO / (S_DO + K_AOB))

EFF_NH1 = 0.50 * (1 + spec_endo_decay * SRT) / (SRT * (EFF_NH - spec_endo_decay) - 1) 
    
S_NH = K_NH4 * (1 + (spec_endo_decay * SRT))

growth_AOB_DO = (spec_growth_AOB * S_DO) / (S_DO + K_AOB)

Effluent_NH4_N_numenator = (SRT * (growth_AOB_DO - spec_endo_decay)) - 1.0

Effluent_NH4_N = (S_NH / Effluent_NH4_N_numenator)

# STEP 19: Calculation of Nitrate production

r_NH = (spec_growth_AOB / 0.15 ) * (IN_NH3 / ( IN_NH3 + K_NH4))

r_NH1 = r_NH * (S_DO / (S_DO + K_AOB)) * 20

NOX = r_NH1 * 1.1 

X_AOB = Q * 0.15 * NOX * 1.1 

X_AOB1 = X_AOB / (V * (1 + spec_endo_decay * SRT))

# STEP 20: Calculation of biomass production

BOD5_removal1 = Influent_BOD1 - Effluent_BOD5

bh_t = bh * (1.04 ** (T - 20))

Px_bio1 = ( Q * YH * BOD5_removal1 ) / 1000

Px_bio2 = [1 + bh_t * 1.1 ]

Px_bio3 = Px_bio1 / Px_bio2

Px_bio4 = (fd_o * bh_t * Q * YH * BOD5_removal1 * 1.1) / 1000

Px_bio5 = 1 + bh_t * 1.1

Px_bio6 = Px_bio4 / Px_bio5

Px_bioi = Px_bio3  + Px_bio6

Px_NH4 = (Q * 0.15 * (NOX)) / 1000

Px_NH41 = 1 + (0.315) * 1.1 

Px_NH42 = Px_NH4 / Px_NH41

Px_bio = (Px_bio3  + Px_bio6) / 0.80                            #kg/d      

TSS_pro1 = Px_bioi + 3421.845

TSS_pro1A = TSS_pro1 

TSS_pro2A = (TSS_pro1 * 1000000 )/ (Q * 1000 ) + TSSi 

Px_bio_mg =  (Px_bio * 1000000 )/ (Q * 1000 ) #mg/L

try12 = Px_bio_mg + TSSi

#STEP 21: Percentage of suspended solids that settles down

Settled_solids =  ((Benchmark_efficiency  / 100) * try12)

Settling_solids = (Settled_solids / 100) * TSS_pro2A

# STEP 22: Effluent TSS 

Eff_TSS =  try12 - Settled_solids

Total_VSS_1A = Eff_TSS * 0.8

settled_solids_isr = TSSi + bio_growth - Px_bio_mg

# STEP 23: Estimated soluble BOD feedback per mg of TSS settled

SBOD_feedback_ratio = 0.3

BOD_fb = SBOD_feedback_ratio * Settled_solids


# STEP 24: Stoichiometric ratio between the uptake of phosphorus (P) and the growth of microbial biomass

aerobic_biomass_yield = 0.5


Biomass_Growth_isr = aerobic_biomass_yield * (Influent_BOD - Trial_new)  # mg/L

P_Uptake_ratio = 2.2

Growth_P_Uptake_isr = (P_Uptake_ratio / 115) * Biomass_Growth_isr  # mgP/L

#STEP 25:  P uptake rate

Px_mg = 0.015 * (Px_bio * 1000)   #g P/ d

Px_g_per_d_1A = ( Px_mg * 1000 ) / ((V * 1000) * IN_PO4)

Px_mg1 = Px_mg / Q                  # g/m3

Effluent_P = IN_PO4 - Px_mg1    #g/m3


#STEP 26: po4 feedback modeling

P_waste_sludge = (try12 * (Q * 1000)) / 1000       #g/d

P_waste_sludge1 = Effluent_P * Q                    #g/d

P_waste_sludge_percentage = (P_waste_sludge1 / P_waste_sludge) * 100  #percentage


P_waste_sludge_mg_l = (P_waste_sludge * 1000) /  (Q * 1000)

P_waste_sludge_final = (P_waste_sludge_percentage / 100) * P_waste_sludge_mg_l

P_waste_settled_solids_1A = (P_waste_sludge_final / try12) * Settled_solids 
                  

A1 = ((1 * 696050000 * IN_PO4) + (0.40 * 1 * 696050000 * IN_PO4)) / (0.20 * 696050000)

A2 = A1 + IN_PO4 

A3 = (0.2 * 696050000 * (A2 - IN_PO4)) / 1000000  #g/d

A4 = A3 * 1000                             #mg/d

A5 = A4 / (Q * 1000)

A6 = A5 * (Settled_solids * 0.8)

EFF_PO4 = Effluent_P + A6 


# STEP 27:  Calculation of NH3 feedback

biomass_growth = 0.5 * (Influent_BOD1 - Effluent_BOD5)

biomass_growth_array = np.full_like(IN_NH3, biomass_growth)

Uptake_N1 = (14 / 115 ) * biomass_growth

NH4_N_feedback_1 = (Uptake_N1 - IN_NH3 + Effluent_NH4_N) 

eff_nh3 = (Uptake_N1 - NH4_N_feedback_1) + IN_NH3

eff_nh3_isr = (Uptake_N1 -NH4_N_feedback_1) + IN_NH3

Removal_N = Uptake_N1 - NH4_N_feedback_1

final_nh3_fb = (Settled_solids * 0.8 ) * (0.0593 / np.where(T > 25, 1, 2))

final_nh3 = Uptake_N1 - final_nh3_fb

KB = 0.08 * (KBTHETA ** (T - 20))

NH3_feedback_T = ((YBNSD * KB * 69605) / V) * 24 

eff_fb_nh3 = NH3_feedback_T + Effluent_NH4_N


import numpy as np

TSS_BOD = 1 - np.exp(-0.10 * Hydrau_Reten)


TSS_BOD1 = Eff_TSS * 0.5 * TSS_BOD

# STEP 28: Calculation of carbonaceous biochemical oxygen demand

Soluble_BOD = Trial_new 



CBOD5 = Soluble_BOD + TSS_BOD1


Flow_rate_2 = Q * 0.000409  # ft3/s

Flow_rate_3 = Flow_rate_2 * 0.646317


# STEP 29: Calculation of Ultimate oxygen demand

UOD_2 = ((cBOD5_Multiplier * CBOD5) + 4.57 * final_nh3_fb) * Flow_rate_3 * 8.34


# Aerated Stabilization Basin 2

# STEP 1: Power supply

Px_1B = aeration_1B / Volume_1B

Power_1B = Px_1B * Volume_1B

#STEP 2: Effluent BOD5 calculation

K1 = 2.5 * (EFF_PO4 / (0.053 + EFF_PO4)) * (theta_T1_B ** (T_1B - 20))

Effluent_BOD51 = CBOD5

Effluent_BOD5_1B = 1 / (1 + (K1 * Hydrau_Reten)) * Effluent_BOD51    


BOD5_removal_1B = CBOD5 - Effluent_BOD5_1B    #mg/


x_numerator_1B = Y * (CBOD5 - Effluent_BOD5_1B)

x_denomenator_1B = 1 + (kd * Hydrau_Reten)

x_1B = x_numerator_1B / x_denomenator_1B  # mg/L

Px_1B = (x_1B * Q) / 1000  # kg/d

Px_O2_1B = 1.42 * Px_1B

O2_requirement_1Bisr = Q * (BOD5_removal_1B / (f * 1000))-(Px_O2_1B)  # kg/d

oxygen_supply1B = ((aeration_1B * 24) / 1000) * Coeff_Power

BOD_Effluent_1B = ((oxygen_supply1B -  Px_O2_1B) / Q ) * (f * 1000)

BOD5_Effluent_1B = CBOD5 - Effluent_BOD5_1B

BOD5_Effluentnp_1B = np.percentile(BOD5_Effluent_1B, 50)


x_numerator_1B = Y * (BOD5_removal - Effluent_BOD5_1B)

x_denomenator_1B = 1 + (kd * Hydrau_Reten)

x_1B = x_numerator_1B / x_denomenator_1B  # mg/L

Px_1B = (x_1B * Q) / 1000  # kg/d

Px_O2_1B = 1.42 * Px_1B

O2_requirement_1B = Q * (BOD5_removal_1B / (f * 1000))-(Px_O2_1B)  # kg/d


Power_req_bod_1B = O2_requirement_1B / (Coeff_Power)  # kW


# STEP 3: Calculation of oxygen transfer rate under standard conditions

pressure = -(g * M * (elevation_a - 0)) / (R * (273.15 + 12))

relative_pressure = math_exp ** pressure

C_20 = 1 + d_e * (Diffuser_depth / Pa)

Oxygen_Con_20 = C20 * C_20

O2_requirement_h1 = oxygen_supply1B / 24

SOTR1_isr = O2_requirement_h1 / (0.50 * Foul_factor)

SOTR2_isr = (beta * (C20 / C12) * relative_pressure * Oxygen_Con_20) - 2.0

SOTR3_isr = Oxygen_Con_20 / SOTR2_isr

SOTR4_isr = SOTR3_isr * (1.024) ** (20 - T_1B)

SOTR5_isr = SOTR1_isr * SOTR4_isr                       #kg O2/ h

SOTR6_isr0 = SOTR5_isr * 26.4                        #kg O2/ d

# STEP 4: For oxygen requirement

O2_requirement_h_isr1 = O2_requirement_1Bisr / 24

SOTR1_isr1 = O2_requirement_h_isr1 / (0.50 * Foul_factor)

SOTR2_isr1 = (beta * (C20 / C12) * relative_pressure * Oxygen_Con_20) - 2.0

SOTR3_isr1 = Oxygen_Con_20 / SOTR2_isr1

SOTR4_isr1 = SOTR3_isr1 * (1.024) ** (20 - T_1B)

SOTR5_isr1 = SOTR1_isr1 * SOTR4_isr1                       #kg O2/ h

SOTR6_isr1 = SOTR5_isr1 * 26.4                        #kg O2/ d

K_isr_0 = (EFF_PO4 / (0.053 + EFF_PO4)) * (theta_T1_B ** (T_1B - 20))

Effluent_BOD_trials2_isr = 1 + (K_isr_0 * Hydrau_Reten) * (SOTR6_isr0 / SOTR6_isr1) 

Trial_new1 = CBOD5 / Effluent_BOD_trials2_isr

#STEP 5: Biomass production


bh_t = bh * (1.04 ** (T_1B - 20))

Px_bio1B = ( Q * YH * (CBOD5 - Effluent_BOD5_1B )) / 1000

Px_bio2B = (1 + bh_t * Hydrau_Reten)

Px_bio3B = Px_bio1B / Px_bio2B

Px_bio4B = (fd_o * bh_t * Q * YH * (CBOD5 - Effluent_BOD5_1B ) * Hydrau_Reten) / 1000

Px_bio5B = 1 + bh_t * Hydrau_Reten 

Px_bio6B = Px_bio4B / Px_bio5B

Px_bioB = Px_bio3B + Px_bio6B

#STEP 6: Temperature correction maximum specific growth

spec_growth_AOB1 = max_growth_coefficient * (1.072 ** (T_1B - 20))

# STEP 7: Temperature correction endogenous decay coefficient 

spec_endo_decay1 = b_20 * (1.029 ** (T_1B - 20))

# STEP 8: Specific growth rate of ammonia oxidizing bacteria 

growth_NH41 = spec_growth_AOB1 * (S_NH3  / (S_NH3  + K_NH4))

spec_growth_NH41 = growth_NH41 * (S_DO / (S_DO + K_AOB)) - spec_endo_decay1

# STEP 9: Solids retention time

SRT1 = ( 1 / spec_growth_NH41) * 1.5 

# STEP 10: Effluent NH3 Calculation

EFF_NH_ISR = spec_growth_AOB1 * (S_DO1 / (S_DO1 + K_AOB))

EFF_NH1_ISR = eff_nh3_isr * (1 + spec_endo_decay1 * SRT1) / (SRT1 * (EFF_NH_ISR - spec_endo_decay1) - 1) 

S_NH1 = K_NH4 * (1 + (spec_endo_decay * SRT1))

growth_AOB_DO1 = spec_growth_AOB1 * ((S_DO) / (S_DO + K_AOB))

Effluent_NH4_N_numenator1 = (SRT1 * (growth_AOB_DO1 - spec_endo_decay1)) - 1.0

Effluent_NH4_N1 = (S_NH1 / Effluent_NH4_N_numenator1 )

# STEP 11: Calculation of Nitrate production

r_NH = (spec_growth_AOB / 0.15 ) * (eff_nh3 / (eff_nh3 + K_NH4))

r_NH12 = r_NH * (S_DO / (S_DO + K_AOB)) * 30

NOX1 = r_NH12 * 1.1 

X_AOB = Q * 0.15 * NOX * SRT 

X_AOB1 = X_AOB / (V * (1 + spec_endo_decay * SRT))

Px_NH4 = (Q * 0.15 * (NOX)) / 1000

Px_NH41 = 1 + (0.315) * Hydrau_Reten 

Px_NH42 = Px_NH4 / Px_NH41

#STEP 12: Calculation of biomass production

Px_bio_isr = (Px_bio3B  + Px_bio6B ) / 0.80

Px_bio_mg1 =  (Px_bio_isr * 1000000 )/ (Q * 1000 )     #mg/L

TSS_pro = Px_bio_isr + ((Q * (Eff_TSS - Total_VSS_1A)) / 1000)

TSS_pro1isr = TSS_pro 

TSS_pro2 = (TSS_pro1isr * 1000000 )/ (Q * 1000 ) + Eff_TSS

Benchmark_efficiency1 = 84 - (10.6 * (aeration_1B / Volume_1B))

TRY1 = Px_bio_mg1 + Total_VSS_1A

# STEP 13: Calculation of solids settled 

Settled_solids1B =  (( TRY1 / 100) * Benchmark_efficiency1)

Settling_solids_1B = (Settled_solids1B / 100) * TSS_pro2

# STEP 14: Effluent TSS

Eff_TSS_1B =  TRY1 - Settled_solids1B


# STEP 15:  Calculation of N Uptake 


biomass_growth1 = 0.5 * (CBOD5 - Effluent_BOD5_1B )

Settled_solids1 = Eff_TSS + biomass_growth1 - (TSS_pro1 * 1000000 )/ (Q * 1000 )

BOD_fb1 = 0.3 * Settled_solids1B

biomass_growth_array1 = np.full_like(IN_NH3, biomass_growth1)

Uptake_N12_isr = (14 / 115 ) * biomass_growth1

# STEP 16: Calculation of NH3 feedback 

NH4_N_feedback_ISR = (Uptake_N12_isr - eff_nh3 + Effluent_NH4_N1) / 1

eff_nh31 = (Uptake_N12_isr - NH4_N_feedback_ISR) + eff_nh3

effluent_nh3_con = NH4_N_feedback_ISR + Effluent_NH4_N1

final_nh3_fb_isr = (Settled_solids1B * 0.8 ) * (0.0593 /  np.where(T > 25, 1, 2))

final_nh3_isr = final_nh3_fb_isr + Effluent_NH4_N1





# STEP 17: Stoichiometric ratio between the uptake of phosphorus (P) and the growth of microbial biomass

P_Uptake_ratio = 2.2

Growth_P_Uptake1 = (P_Uptake_ratio / 115) * biomass_growth1 # mgP/L



# STEP 18: Phosphate uptake rate

Px_mg_isr = 0.015 * (Px_bio_isr * 1000)   #g P/ d

Px_g_per_d = ( Px_mg_isr * 1000 ) / ((V * 1000) * EFF_PO4)   # per day

Px_mg1_isr = Px_mg_isr / Q                  # g/m3

Effluent_P_isr = EFF_PO4 - Px_mg1_isr    #g/m3

PO4_fb_isr = (Settled_solids1B * 0.8 )  * (0.0049 / 1)



# STEP 19: po4 feedback modeling

P_waste_sludge_1B = (TRY1 * (Q * 1000)) / 1000       #g/d

P_waste_sludge1_1B = Effluent_P_isr * Q                    #g/d

P_waste_sludge_percentage_1B = (P_waste_sludge1_1B / P_waste_sludge_1B) * 100  #percentage

P_waste_sludge_mg_l_1B = (P_waste_sludge_1B * 1000) /  (Q * 1000)

P_waste_sludge_final_1B = (P_waste_sludge_percentage_1B / 100) * P_waste_sludge_mg_l_1B

P_waste_settled_solids = (P_waste_sludge_final_1B / TRY1) * Settled_solids1B
                   

A1_isr = ((1 * 586790000 * EFF_PO4) + (0.40 * 1 * 586790000 * EFF_PO4)) / (0.20 * 586790000)

A2_isr = A1_isr + EFF_PO4

A3_isr = (0.2 * 586790000 * (A2_isr - EFF_PO4)) / 1000000  #g/d

A4_isr = A3_isr * 1000                             #mg/d

A5_isr = A4_isr / (Q * 1000)

A6_isr = A5_isr * (Settled_solids1B * 0.8) 

#STEP 20: Effluent PO4

EFF_PO4_isr = Effluent_P_isr + A6_isr

import numpy as np

TSS_BODisr = 1 - np.exp(-0.10 * Hydrau_Reten)

# STEP 21: Carbonaceous biochemcial oxygen demand

TSS_BOD1isr = Eff_TSS_1B * 0.5 * TSS_BODisr



Soluble_BOD1 =  Trial_new1 



CBOD51 = Soluble_BOD1  + TSS_BOD1isr

# STEP 22: Ultimate oxygen demand

UOD_3 = ((cBOD5_Multiplier * CBOD51) + 4.57 * final_nh3_fb_isr) * Flow_rate_3 * 8.34








