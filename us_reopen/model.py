"""US county reopening simulation model."""

from us_reopen.us_data import getInformation
from us_reopen.interventions import (
    apply_daily_mobility_intervention,
    process_number_to_fns,
    shutdown_county_mask,
)
from us_reopen.config import (
    AGE_BETA,
    AGES,
    ALPHA_OFFSET,
    ALPHAMAT,
    CONFIRMED_PROCESS_METHODS,
    CONTACTABLE_STATUS_IDS,
    DEFAULT_WEEKCOUNT,
    DRATIO_BASE,
    E_BETA,
    E_DAYS_PROB,
    FLOW_RATIO,
    H2DRATIO_BASE,
    H_DAYS_PROB,
    HRATIO_BASE,
    IA_BETA,
    IA_RATIO,
    IA_TEST,
    I_DAYS_PROB,
    I_TEST,
    I_TEST_STAGE1,
    INFECTABLE_STATUS_IDS,
    MOVEABLE_STATUS_IDS,
    RECOVER_DATE,
    RECOVER_NAME,
    RECOVER_RATE,
    RMAX,
    S2E_VERSION,
    STATUS,
    STATUS_BETA,
    STATUS_INDEX,
    TEST_OVER,
    get_h2d_ratio,
)
from us_reopen.immunity import getR
from us_reopen.initialization import load_csa_county_indices, select_initial_infections
from us_reopen.transmission import (
    build_county_transmission,
    load_early_reopen_flags,
    make_pcf_beta_fn,
    update_county_transmission_for_day,
)
from us_reopen.model_utils import (
    DrawMapGif,
    MapInit,
    addSelected_manual,
    initMapGif,
    process_kwargs,
    saveMapGif_mpa,
)
from us_reopen.od_mobility import process_sample_od
from us_reopen.population import buildnp
from us_reopen.state_initialization import build_initial_state
from us_reopen.cli_options import (
    build_subdir,
    build_suffix,
    parse_cli_args,
    sanitize_run_label,
)
from us_reopen.experiment_modes import (
    apply_args_and_kwargs,
    run_curve_experiment,
    run_experiment_batch as _run_experiment_batch,
    run_multiple_experiment,
    run_single_experiment,
    select_process_functions as _select_process_functions,
)
from us_reopen.disease_step import advance_disease_step
from us_reopen.randomness import initialize_numpy_random
from us_reopen.scalar_outputs import write_scalar_outputs
from us_reopen.simulation_metrics import SimulationMetrics

import numpy as np
import itertools
weekcount = DEFAULT_WEEKCOUNT
status2id = STATUS_INDEX
moveable = MOVEABLE_STATUS_IDS
contactable = CONTACTABLE_STATUS_IDS
E_beta = E_BETA
Ia_beta = IA_BETA
Ia_ratio = IA_RATIO
statusbeta = STATUS_BETA
E_days_prob = E_DAYS_PROB
I_days_prob = I_DAYS_PROB
H_days_prob = H_DAYS_PROB
TestOver = TEST_OVER
alphaoffset = ALPHA_OFFSET
I_test_stage1 = I_TEST_STAGE1
I_test = I_TEST
Ia_test = IA_TEST
Dratio_base = DRATIO_BASE
H2Dratio_base = H2DRATIO_BASE
Hratio_base = HRATIO_BASE
get_H2Dratio = get_h2d_ratio
flow_ratio = FLOW_RATIO
agebeta = AGE_BETA
ages = AGES
alphamat = ALPHAMAT


def _resolve_process_inf(process_inf, suffix):
    """Resolve CLI behavior while supporting the direct-call interface."""
    if process_inf is None:
        return "_inf" in suffix
    return bool(process_inf)


def ODseir_simple(*args, **kwargs):
    """Run one county-level simulation using the keyword API."""
    townIDs,town_population,town_area,_,_,ageprops,REF_POP_DENSITY = kwargs['csainfo']
    Rtype = kwargs['Rtype']
    R_days_prob = getR(Rtype,Rmax = RMAX) 
    Pratio_num = kwargs['Pratio']
    subdir_name  = kwargs.get("subdir_name","")
    initpopulations = np.array(list(town_population.values()))
    run_weekcount = kwargs.get("weekcount", weekcount)
    rng = initialize_numpy_random(kwargs.get("seed", 0))
    flow_ratio_local,dryrun ,gif ,scalar ,_suffix, beta_density  = process_kwargs(**kwargs)
    alpha_ratio = kwargs.get("alpha_ratio",1.0)
    process_threshold_date,process_method = kwargs.get("process_inequal",(125,0))

    START_AT = select_initial_infections(
        kwargs["init_method"],
        initpopulations,
        townIDs,
        beta_density,
        TestOver,
    )
    csa2county, csa2countyidx = load_csa_county_indices(townIDs)
    print("suffix:",_suffix,flush = True)
    if flow_ratio_local is None:
        flow_ratio_local = flow_ratio
    
    _vmax, _, map_flag = MapInit()
    beta = np.array(agebeta)
    county_transmission = build_county_transmission(
        beta_density,
        kwargs["period"],
        townIDs,
        alphamat,
        alphaoffset,
        TestOver,
    )
    countyalpha = county_transmission.countyalpha
    alphamat_county = county_transmission.alphamat_county
    xmax_date = county_transmission.xmax_date
    flag_early_reopen = load_early_reopen_flags(townIDs, RECOVER_DATE)
    
    infectables = INFECTABLE_STATUS_IDS
    initial_state = build_initial_state(
        townIDs,
        initpopulations,
        ageprops,
        ages,
        STATUS,
        status2id,
        START_AT,
        RMAX,
        rng,
    )
    townsList = initial_state.towns_list
    E_date = initial_state.e_date
    I_date = initial_state.i_date
    It_date = initial_state.it_date
    Ia_date = initial_state.ia_date
    H_date = initial_state.h_date
    R_date = initial_state.r_date
    
    # Direct calls infer the setting from the suffix when no flag is provided.
    process_inf = _resolve_process_inf(kwargs.get("process_inf"), _suffix)
    process_period = 100000 if process_inf else 60
    get_pcf_beta = make_pcf_beta_fn(
        beta_density,
        countyalpha,
        RECOVER_RATE,
        town_area,
        REF_POP_DENSITY,
    )
    if not dryrun:
        if isinstance( kwargs['sampleOD'] ,tuple):
            two_OD_mat_flag = True
            sampleOD_workday,sampleOD_weekend  = kwargs['sampleOD'] 
            sampleOD_workday = process_sample_od(sampleOD_workday, kwargs['period'], initpopulations, flow_ratio_local)
            sampleOD_weekend = process_sample_od(sampleOD_weekend, kwargs['period'], initpopulations, flow_ratio_local)
        else:
            two_OD_mat_flag = False
            sampleOD  = process_sample_od(kwargs['sampleOD'], kwargs['period'], initpopulations, flow_ratio_local)
        
            
        
        
    else:
        two_OD_mat_flag  = True
        sampleOD_workday :np.ndarray = rng.integers(0,3,[7*24,len(initpopulations),len(initpopulations)])*flow_ratio_local
        sampleOD_weekend :np.ndarray = rng.integers(0,3,[7*24,len(initpopulations),len(initpopulations)])*flow_ratio_local
    selected = []
    addSelected_manual(selected,townsList,np.argsort(initpopulations)[-6:],initpopulations,town_area)

    metrics = SimulationMetrics(selected, townsList, initpopulations, csa2county, csa2countyidx, START_AT)
    
    towns:np.ndarray = buildnp(townsList,len(ages),len(STATUS)).astype(int)
    
    
    if gif:
        map_gifwriter,_,map_gif_ax,map_gif_cb = initMapGif("output/gifs/I_proportion.mp4")
    

    population = initpopulations.copy()
    if two_OD_mat_flag:
        ODandLeft_workday = sampleOD_workday/sampleOD_workday.sum(axis = 1,keepdims = True)  
        ODandLeft_weekend = sampleOD_weekend/sampleOD_weekend.sum(axis = 1,keepdims = True)  
    else:
        ODandLeft = sampleOD/sampleOD.sum(axis = 1,keepdims = True)  
    for week, day, in itertools.product(range(run_weekcount), range(7)):
        population = towns[:,:,contactable].sum(axis = (1,2))
        if dryrun:
            print(f"Week{week}Day{day} S:{towns[:,:,status2id['S']].sum():.2f} E:{towns[:,:,status2id['E']].sum():.2f} I:{towns[:,:,status2id['I']].sum():.2f} R:{towns[:,:,status2id['R']].sum():.2f} P:{towns[:,:,status2id['P']].sum():.2f}")            
            if gif:
                data = rng.random(len(towns))
                if max(data)>_vmax:
                    _vmax = max(data)
                title = f"week {week:0=2}-day {day:0=2}-I proportion"
                DrawMapGif(map_gifwriter,map_gif_ax,map_gif_cb,data,map_flag,title,vmax = _vmax if _vmax>0.0 else 1.0)
            continue

        metrics.record_population_state(towns, population, status2id)
        if gif:
            data = towns[:,:,[status2id['I']]].sum(axis = (1,2))/towns.sum(axis=(1,2))
            if max(data)>_vmax:
                _vmax = max(data)
            _vmax = max(data)
            title = f"week {week:0=2}-day {day:0=2}-I proportion"
            DrawMapGif(map_gifwriter,map_gif_ax,map_gif_cb,data,map_flag,title,vmin =0.0,vmax = 1.0,log_norm=True)
            
        if two_OD_mat_flag:
            sampleOD = sampleOD_workday if day<=4 else sampleOD_weekend
            ODandLeft = ODandLeft_workday if day<=4 else ODandLeft_weekend
        
        if process_method > 0:
            ODandLeft = apply_daily_mobility_intervention(
                day_index=week * 7 + day,
                process_method=process_method,
                process_threshold_date=process_threshold_date,
                process_period=process_period,
                sample_od=sampleOD,
                od_and_left=ODandLeft,
                initpopulations=initpopulations,
                population=population,
                towns=towns,
                town_ids=townIDs,
                status2id=status2id,
                infectables=infectables,
                statusbeta=statusbeta,
                agebeta=agebeta,
                get_pcf_beta=get_pcf_beta,
                mat_process_func=kwargs["mat_process_func"],
                subdir_name=subdir_name,
                suffix=_suffix,
            )
            alphamat_county = update_county_transmission_for_day(
                day_index=week * 7 + day,
                beta_density=beta_density,
                alphamat=alphamat,
                countyalpha=countyalpha,
                xmax_date=xmax_date,
                flag_early_reopen=flag_early_reopen,
            )
            alpha_ratio = kwargs["alpha_ratio"]


        print("Date:",week*7+day,"processAfterShutdown.py,process = ",process_method,"ODandLeft diag=",np.diag(ODandLeft).sum())
        step = advance_disease_step(
            towns=towns,
            population=population,
            od_and_left=ODandLeft,
            rng=rng,
            s2eversion=kwargs["s2eversion"],
            beta_density=beta_density,
            infectables=infectables,
            statusbeta=statusbeta,
            alphamat=alphamat,
            alphamat_county=alphamat_county,
            beta=beta,
            alpha_ratio=alpha_ratio,
            town_area=town_area,
            ref_pop_density=REF_POP_DENSITY,
            moveable=moveable,
            status2id=status2id,
            e_date=E_date,
            i_date=I_date,
            it_date=It_date,
            ia_date=Ia_date,
            h_date=H_date,
            r_date=R_date,
            e_days_prob=E_days_prob,
            i_days_prob=I_days_prob,
            h_days_prob=H_days_prob,
            r_days_prob=R_days_prob,
            ia_ratio=Ia_ratio,
            ia_test=Ia_test,
            i_test=I_test,
            hratio_base=Hratio_base,
            pratio_num=Pratio_num,
            get_h2d_ratio=get_H2Dratio,
            day_index=week*7+day,
        )
        metrics.record_transition_state(
            week*7+day,
            step.s2e,
            step.s2e_rate,
            step.r2s,
            step.i2r,
            step.i2p,
            step.tested_by_region,
            step.delta_d,
            population,
        )
        metrics.record_delay_profiles(
            E_date,
            I_date,
            Ia_date,
            R_date,
            step.e_date_new,
            step.i_date_new,
            step.ia_date_new,
            step.r_date_new,
            population,
        )
        
    if gif:
        saveMapGif_mpa(map_gifwriter)
    


    if scalar:
        write_scalar_outputs(metrics, townsList, csa2county, initpopulations, subdir_name, _suffix)
    return metrics.result_tuple()

def ODseir_USA(*args, **kwargs):
    return run_single_experiment(ODseir_simple, *args, **kwargs)


def ODseir_compare(*args, **kwargs):
    return run_curve_experiment(ODseir_simple, *args, **kwargs)


def ODseir_multi(expi=None, count=20, a_value=None, a_label=None, *args, **kwargs):
    return run_multiple_experiment(
        ODseir_simple,
        expi,
        count,
        a_value,
        a_label,
        *args,
        **kwargs,
    )

def main(argv=None):
    args, explicit_a_values = parse_cli_args(argv, weekcount)
    alpha_ratio =  float(args.alpha_ratio_name.replace('p','.'))
    flow_ratio =  float(args.flow_ratio_name.replace('p','.'))
    s2eversion = S2E_VERSION
    beta_density = args.beta_density 
    init_method = args.init_method
    run_label = sanitize_run_label(args.run_label)
    for i in [0,]:
        process_inequal = (args.process_threshold,args.process_method+i)
        suffix = build_suffix(args, beta_density, init_method, run_label)
        subdir = build_subdir(args, beta_density, suffix, run_label)
        if args.mode == 'single':
            ODseir_USA(dryrun = False,gif = False,scalar = True,
                    beta_density=beta_density,
                    period = args.period,
                    suffix = suffix,
                    alpha_ratio = alpha_ratio,
                    flow_ratio = flow_ratio,
                    s2eversion=s2eversion,
                    init_method = init_method,
                    process_inequal=process_inequal,
                    process_inf = args.process_inf,
                    subdir = subdir,
                    Rtype = args.Rtype,
                    Pratio = float(args.Pratio.replace("p",".")),
                    weekcount = args.weekcount,
                    seed = args.seed)
        if args.mode == 'curve':
            ODseir_compare(dryrun = False,gif = False,scalar = False,
                    beta_density=beta_density,
                    period = args.period,
                    suffix = suffix,
                    alpha_ratio = alpha_ratio,
                    flow_ratio = flow_ratio,
                    s2eversion=s2eversion,
                    init_method = init_method,
                    process_inequal=process_inequal,
                    process_inf = args.process_inf,
                    subdir = subdir,
                    Rtype = args.Rtype,
                    Pratio = float(args.Pratio.replace("p",".")),
                    curve_points = args.curve_points,
                    a_values = explicit_a_values,
                    poolsize = args.poolsize,
                    process_fn_names = args.process_fn_names,
                    process_fn_limit = args.process_fn_limit,
                    weekcount = args.weekcount,
                    seed = args.seed)
        if args.mode == 'multiple':
            for expi in (args.multiple_idx or []):
                ODseir_multi(
                    expi = expi,
                    count = args.repeat_count,
                    dryrun = False,gif = False,scalar = False,
                        beta_density=beta_density,
                        period = args.period,
                        suffix = suffix,
                        alpha_ratio = alpha_ratio,
                        flow_ratio = flow_ratio,
                        s2eversion=s2eversion,
                        init_method = init_method,
                        process_inequal=process_inequal,
                        process_inf = args.process_inf,
                        subdir = subdir,
                        Rtype = args.Rtype,
                        Pratio = float(args.Pratio.replace("p",".")),
                        multiple_poolsize = args.multiple_poolsize,
                        multiple_grid_points = args.multiple_grid_points,
                        process_fn_names = args.process_fn_names,
                        process_fn_limit = args.process_fn_limit,
                        weekcount = args.weekcount,
                        seed = args.seed)


if __name__ == "__main__":
    main()
