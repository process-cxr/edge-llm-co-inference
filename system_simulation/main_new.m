clc;
clearvars -except optimization_mode random_seed;
close all;

% Scan-friendly entry script for the video-caption split benchmark.
%
% Modes:
% - 'single'        : run one (T0, E0) pair with full verbose logs
% - 'scan_fixed_E0' : fix E0, sweep T0
% - 'scan_fixed_T0' : fix T0, sweep E0
% - 'scan_both'     : run both benchmark lines back-to-back

if ~exist('optimization_mode', 'var') || isempty(optimization_mode)
    optimization_mode = 'joint';
end
if ~exist('random_seed', 'var') || isempty(random_seed)
    random_seed = 42;
end

script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir);
mint_dir = fullfile(script_dir, '..', 'mint');
if exist(mint_dir, 'dir')
    addpath(mint_dir);
end

switch optimization_mode
    case 'joint'
        summary_header = 'BENCHMARK SUMMARY';
        scan_tag_prefix = 'optim_rho_benchmark_';
    case 'fixed_p'
        summary_header = 'FIXED-P BASELINE SUMMARY';
        scan_tag_prefix = 'fixed_p_baseline_';
    case 'round_robin'
        summary_header = 'ROUND-ROBIN BASELINE SUMMARY';
        scan_tag_prefix = 'round_robin_baseline_';
    case 'random_scheduling'
        summary_header = 'RANDOM SCHEDULING BASELINE SUMMARY';
        scan_tag_prefix = sprintf('random_scheduling_seed%d_baseline_', random_seed);
    otherwise
        error('Unsupported optimization_mode: %s', optimization_mode);
end

run_mode = 'scan_both';

% ---------------- Comparison preset ----------------
% 'energy_sensitive' keeps the previous setup that makes fixed-T0 / varying-E0
% react strongly through rho.
% 'bw_sensitive' is recommended when you want Proposed vs Fixed-bandwidth
% to separate more clearly in rho.
comparison_profile = 'bw_sensitive';

% ---------------- Fixed scan lines ----------------
% Keep these benchmark scan lines unchanged across all comparison profiles.
single_T0 = 11.0;
single_E0 = 3.90;

fixed_E0 = 4.00;
sweep_T0_list = [12.2, 11.8, 11.4, 11.0, 10.6, 10.2];

fixed_T0 = 12.2;
sweep_E0_list = [3.97, 3.92, 3.87, 3.82, 3.77, 3.72];

switch comparison_profile
    case 'energy_sensitive'
        device_energy_scale = 8.0;
        total_bandwidth_MHz = 100.0;
        p_max_value = 1.0;
        rho_init_value = 1.0;

    case 'bw_sensitive'
        device_energy_scale = 9.0;
        total_bandwidth_MHz = 40.0;
        p_max_value = 0.6;
        rho_init_value = 1.0;

    otherwise
        error('Unsupported comparison_profile: %s', comparison_profile);
end

% ---------------- Runtime / logging options ----------------
scan_tag = [scan_tag_prefix, comparison_profile];
single_verbose = true;
scan_verbose = false;
rerun_final_milp = true;
save_scan_artifacts = true;
if any(strcmp(optimization_mode, {'round_robin', 'random_scheduling'}))
    rerun_final_milp = false;
end

case_list = build_case_list(run_mode, single_T0, single_E0, fixed_E0, sweep_T0_list, fixed_T0, sweep_E0_list, ...
    single_verbose, scan_verbose, rerun_final_milp, device_energy_scale, rho_init_value, total_bandwidth_MHz, p_max_value, comparison_profile, optimization_mode, random_seed);
shapley_csv = fullfile(script_dir, '..', '..', 'common', 'shapley_outputs', 'video_shapley_weights.csv');
importance_weights = load_shapley_weights(shapley_csv, 8, 3);
for case_idx = 1:numel(case_list)
    case_list(case_idx).importance_weights = importance_weights;
end
num_cases = numel(case_list);

results = struct([]);

for case_idx = 1:num_cases
    cfg = case_list(case_idx);
    fprintf('\n============================================================\n');
    fprintf('Running case %d/%d | %s | mode=%s | T0=%.4f | E0=%.4f\n', ...
        case_idx, num_cases, cfg.label, cfg.optimization_mode, cfg.T_0, cfg.E_0);
    fprintf('============================================================\n');

    case_result = run_case_main_new_gemini(cfg);
    if isempty(results)
        results = case_result;
    else
        results(case_idx) = case_result;
    end
    print_case_digest(results(case_idx), case_idx, num_cases);
end

if num_cases > 1
    fprintf('\n================ %s ================\n', summary_header);
    for case_idx = 1:num_cases
        result = results(case_idx);
        fprintf('%2d. %-18s | T0=%.4f | E0=%.4f | rho_mean=%.4f | rho_min=%.4f | pruned=%s\n', ...
            case_idx, result.label, result.T_0, result.E_0, result.rho_mean, result.rho_min, mat2str(result.pruned_clusters));
        fprintf('    rho_cluster_mean      = [%s]\n', format_numeric_row(result.rho_cluster_mean));
        fprintf('    fixed-schedule exact  = delay %.6f / energy %.6f / feasible %d\n', ...
            result.fixed_schedule_delay, result.fixed_schedule_energy, result.fixed_schedule_feasible);
        fprintf('    reMILP exact          = delay %.6f / energy %.6f / feasible %d\n', ...
            result.rerun_milp_delay, result.rerun_milp_energy, result.rerun_milp_feasible);
    end
end

if num_cases > 1 && save_scan_artifacts
    artifact_info = write_scan_artifacts(results, run_mode, scan_tag);
    fprintf('\n[Save] CSV summary -> %s\n', artifact_info.csv_path);
    fprintf('[Save] TXT log     -> %s\n', artifact_info.txt_path);
    fprintf('[Save] MAT bundle  -> %s\n', artifact_info.mat_path);
end


function case_list = build_case_list(run_mode, single_T0, single_E0, fixed_E0, sweep_T0_list, fixed_T0, sweep_E0_list, ...
    single_verbose, scan_verbose, rerun_final_milp, device_energy_scale, rho_init_value, total_bandwidth_MHz, p_max_value, comparison_profile, optimization_mode, random_seed)
case_list = struct([]);

switch run_mode
    case 'single'
        case_list = make_case_cfg('single', 'single', single_T0, single_E0, single_verbose, rerun_final_milp, ...
            device_energy_scale, rho_init_value, total_bandwidth_MHz, p_max_value, comparison_profile, optimization_mode, random_seed);

    case 'scan_fixed_E0'
        case_list = repmat(make_case_cfg('', '', 0, 0, scan_verbose, rerun_final_milp, ...
            device_energy_scale, rho_init_value, total_bandwidth_MHz, p_max_value, comparison_profile, optimization_mode, random_seed), numel(sweep_T0_list), 1);
        for idx = 1:numel(sweep_T0_list)
            case_list(idx) = make_case_cfg(sprintf('fixE0_%02d', idx), 'fixed_E0', sweep_T0_list(idx), fixed_E0, scan_verbose, rerun_final_milp, ...
                device_energy_scale, rho_init_value, total_bandwidth_MHz, p_max_value, comparison_profile, optimization_mode, random_seed);
        end

    case 'scan_fixed_T0'
        case_list = repmat(make_case_cfg('', '', 0, 0, scan_verbose, rerun_final_milp, ...
            device_energy_scale, rho_init_value, total_bandwidth_MHz, p_max_value, comparison_profile, optimization_mode, random_seed), numel(sweep_E0_list), 1);
        for idx = 1:numel(sweep_E0_list)
            case_list(idx) = make_case_cfg(sprintf('fixT0_%02d', idx), 'fixed_T0', fixed_T0, sweep_E0_list(idx), scan_verbose, rerun_final_milp, ...
                device_energy_scale, rho_init_value, total_bandwidth_MHz, p_max_value, comparison_profile, optimization_mode, random_seed);
        end

    case 'scan_both'
        num_T = numel(sweep_T0_list);
        num_E = numel(sweep_E0_list);
        case_list = repmat(make_case_cfg('', '', 0, 0, scan_verbose, rerun_final_milp, ...
            device_energy_scale, rho_init_value, total_bandwidth_MHz, p_max_value, comparison_profile, optimization_mode, random_seed), num_T + num_E, 1);
        for idx = 1:num_T
            case_list(idx) = make_case_cfg(sprintf('fixE0_%02d', idx), 'fixed_E0', sweep_T0_list(idx), fixed_E0, scan_verbose, rerun_final_milp, ...
                device_energy_scale, rho_init_value, total_bandwidth_MHz, p_max_value, comparison_profile, optimization_mode, random_seed);
        end
        for idx = 1:num_E
            case_list(num_T + idx) = make_case_cfg(sprintf('fixT0_%02d', idx), 'fixed_T0', fixed_T0, sweep_E0_list(idx), scan_verbose, rerun_final_milp, ...
                device_energy_scale, rho_init_value, total_bandwidth_MHz, p_max_value, comparison_profile, optimization_mode, random_seed);
        end

    otherwise
        error('Unsupported run_mode: %s', run_mode);
end
end


function cfg = make_case_cfg(label, family, T_0, E_0, verbose, rerun_final_milp, device_energy_scale, rho_init_value, total_bandwidth_MHz, p_max_value, comparison_profile, optimization_mode, random_seed)
cfg = struct( ...
    'label', label, ...
    'family', family, ...
    'T_0', T_0, ...
    'E_0', E_0, ...
    'verbose', verbose, ...
    'rerun_final_milp', rerun_final_milp, ...
    'device_energy_scale', device_energy_scale, ...
    'total_bandwidth_MHz', total_bandwidth_MHz, ...
    'p_max_value', p_max_value, ...
    'p_fixed_value', p_max_value, ...
    'comparison_profile', comparison_profile, ...
    'optimization_mode', optimization_mode, ...
    'random_seed', random_seed, ...
    'rho_init', rho_init_value * ones(8, 3) ...
);
end


function print_case_digest(result, case_idx, num_cases)
fprintf('\n---- Case digest %d/%d ----\n', case_idx, num_cases);
fprintf('label                    : %s\n', result.label);
fprintf('family                   : %s\n', result.family);
fprintf('optimization_mode        : %s\n', result.optimization_mode);
fprintf('outer iterations         : %d\n', result.outer_iter);
fprintf('last cvx status          : %s\n', result.last_cvx_status);
fprintf('last cvx objective       : %.6f\n', result.last_cvx_objective);
fprintf('rho_mean / rho_min / rho_max : %.6f / %.6f / %.6f\n', result.rho_mean, result.rho_min, result.rho_max);
fprintf('pruned clusters          : %s\n', mat2str(result.pruned_clusters));
fprintf('rho_cluster_mean         : [%s]\n', format_numeric_row(result.rho_cluster_mean));
fprintf('fixed schedule exact     : delay %.6f | energy %.6f | feasible %d\n', ...
    result.fixed_schedule_delay, result.fixed_schedule_energy, result.fixed_schedule_feasible);
fprintf('reMILP exact             : delay %.6f | energy %.6f | feasible %d\n', ...
    result.rerun_milp_delay, result.rerun_milp_energy, result.rerun_milp_feasible);
end


function artifact_info = write_scan_artifacts(results, run_mode, scan_tag)
script_dir = fileparts(mfilename('fullpath'));
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
out_dir = fullfile(script_dir, '..', '..', 'outputs', 'optimization', 'scans', ...
    sprintf('%s_%s_%s', scan_tag, run_mode, timestamp));
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

csv_path = fullfile(out_dir, 'scan_summary.csv');
txt_path = fullfile(out_dir, 'scan_summary.txt');
mat_path = fullfile(out_dir, 'scan_results.mat');

num_cases = numel(results);
labels = cell(num_cases, 1);
families = cell(num_cases, 1);
optimization_modes = cell(num_cases, 1);
schedule_modes = cell(num_cases, 1);
statuses = cell(num_cases, 1);
rho_cluster_strings = cell(num_cases, 1);
pruned_cluster_strings = cell(num_cases, 1);
random_seeds = NaN(num_cases, 1);

for idx = 1:num_cases
    labels{idx} = results(idx).label;
    families{idx} = results(idx).family;
    optimization_modes{idx} = results(idx).optimization_mode;
    schedule_modes{idx} = results(idx).schedule_mode;
    statuses{idx} = results(idx).last_cvx_status;
    rho_cluster_strings{idx} = ['[', format_numeric_row(results(idx).rho_cluster_mean), ']'];
    pruned_cluster_strings{idx} = mat2str(results(idx).pruned_clusters);
    random_seeds(idx) = results(idx).random_seed;
end

summary_table = table( ...
    (1:num_cases)', ...
    labels, ...
    families, ...
    optimization_modes, ...
    schedule_modes, ...
    random_seeds, ...
    [results.T_0]', ...
    [results.E_0]', ...
    [results.outer_iter]', ...
    [results.last_cvx_objective]', ...
    statuses, ...
    [results.rho_mean]', ...
    [results.rho_min]', ...
    [results.rho_max]', ...
    [results.num_pruned_entries]', ...
    rho_cluster_strings, ...
    pruned_cluster_strings, ...
    [results.fixed_schedule_delay]', ...
    [results.fixed_schedule_energy]', ...
    [results.fixed_schedule_feasible]', ...
    [results.rerun_milp_delay]', ...
    [results.rerun_milp_energy]', ...
    [results.rerun_milp_feasible]', ...
    'VariableNames', { ...
        'case_id', 'label', 'family', 'optimization_mode', 'schedule_mode', 'random_seed', ...
        'T0', 'E0', 'outer_iter', 'last_cvx_objective', 'last_cvx_status', ...
        'rho_mean', 'rho_min', 'rho_max', 'num_pruned_entries', 'rho_cluster_mean', 'pruned_clusters', ...
        'fixed_schedule_delay', 'fixed_schedule_energy', 'fixed_schedule_feasible', ...
        'rerun_milp_delay', 'rerun_milp_energy', 'rerun_milp_feasible' ...
    });

writetable(summary_table, csv_path);

fid = fopen(txt_path, 'w');
fprintf(fid, 'run_mode = %s\n', run_mode);
fprintf(fid, 'generated_at = %s\n\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
for idx = 1:num_cases
    result = results(idx);
    fprintf(fid, 'Case %02d | %s | family=%s | mode=%s | schedule=%s | seed=%d | T0=%.4f | E0=%.4f\n', ...
        idx, result.label, result.family, result.optimization_mode, result.schedule_mode, result.random_seed, result.T_0, result.E_0);
    fprintf(fid, '  outer_iter            = %d\n', result.outer_iter);
    fprintf(fid, '  last_cvx_status       = %s\n', result.last_cvx_status);
    fprintf(fid, '  last_cvx_objective    = %.6f\n', result.last_cvx_objective);
    fprintf(fid, '  rho_cluster_mean      = [%s]\n', format_numeric_row(result.rho_cluster_mean));
    fprintf(fid, '  rho_min/max/mean      = %.6f / %.6f / %.6f\n', result.rho_min, result.rho_max, result.rho_mean);
    fprintf(fid, '  pruned_clusters       = %s\n', mat2str(result.pruned_clusters));
    fprintf(fid, '  fixed_schedule_exact  = delay %.6f | energy %.6f | feasible %d\n', ...
        result.fixed_schedule_delay, result.fixed_schedule_energy, result.fixed_schedule_feasible);
    fprintf(fid, '  reMILP_exact          = delay %.6f | energy %.6f | feasible %d\n', ...
        result.rerun_milp_delay, result.rerun_milp_energy, result.rerun_milp_feasible);
    fprintf(fid, '  rho_final =\n');
    for row = 1:size(result.rho_final, 1)
        fprintf(fid, '    %s\n', format_numeric_row(result.rho_final(row, :)));
    end
    fprintf(fid, '\n');
end
fclose(fid);

save(mat_path, 'results', 'summary_table');

artifact_info = struct('out_dir', out_dir, 'csv_path', csv_path, 'txt_path', txt_path, 'mat_path', mat_path);
end


function text = format_numeric_row(values)
parts = arrayfun(@(value) sprintf('%.4f', value), values, 'UniformOutput', false);
text = strjoin(parts, ', ');
end
