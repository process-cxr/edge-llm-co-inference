clc; clear; close all;
% 基线：简单轮询调度（不优化 rho/p），用于与联合优化方案对比。

%% ---------------- 基本尺寸 ----------------
M  = 8;                 % 簇数
Kc = 3;                 % 每簇用户数
U  = M*Kc;              % 总用户数
G  = 2;                 % GPU 个数
Q  = M;                 % 每 GPU 的槽位上界（取 M 足够）

%% ---------------- 系统参数（ ----------------
Nflop_server = [4,8,30,20,400,13,5000,16];   % GFlops/任务
cycle_server = 128;

% GPU 
f_g   = [1.00, 0.85];                    % GHz
phi_g = [0.10, 0.12];                    % 能耗系数
c_g   = [128, 140];                      % 每周期 FLOPs（或等效参数）
eta_server = 2;
P_b        = 0;                          % 空闲功率（可改）

% 设备
f_device     = 0.3;                      % GHz
Nflop_device = 4;
phi_device   = 0.5e-1;
cycle_device = 32;
eta_device   = 1;

b = 32;
theta = (768*b*2);                       % 每设备上传比特数


dis=[500,400,600;
     200,220,150;
     700,730,780;
     50, 80, 30;
     1000,1200,900;
     2000,2100,1900;
     900,780,920;
     550,620,490];                        
noise_enlarge = 1e10;
h = 1e-3 * (dis.^(-2.8)) * noise_enlarge; 
P_max = 1 * ones(1,U);                    
B     = 5e6;
N_0   = 2e-19 * noise_enlarge;
noise = B * N_0;

%% -------- 计算簇到达时刻 --------
rho_mk = ones(M,Kc);
g_vec   = reshape(h.', 1, []);            
rho_vec = reshape(rho_mk.', 1, []);       
p_vec   = P_max;                          

tI_vec = (rho_vec .* Nflop_device) / (f_device * cycle_device);
R_vec  = B .* log2(1 + g_vec .* p_vec ./ noise);
tU_vec = theta ./ R_vec;
tA_vec = tI_vec + tU_vec;                 

tA_mat = reshape(tA_vec, Kc, M).';        
tR_m   = max(tA_mat, [], 2)';             

%% -------- 服务时长能耗系数 --------
S_mg = zeros(M,G);   
Wmg  = zeros(M,G);   
for g = 1:G
    S_mg(:,g) = Nflop_server(:) / (f_g(g) * c_g(g));
    Wmg(:,g)  = eta_server * phi_g(g) * (f_g(g)^2) * (Nflop_server(:) / c_g(g));
end

%% Round Robin
[~, order] = sort(tR_m, 'ascend');   

x_rr  = zeros(M, G, Q);              
T_rr  = zeros(G, Q);                 
lastT = zeros(G, 1);                 
qpos  = zeros(G, 1);                 

for i = 1:M
    m = order(i);                    
    g = mod(i-1, G) + 1;             
    qpos(g) = qpos(g) + 1;
    q = qpos(g);
    assert(q <= Q, 'Q 不够大，GPU %d 的槽位已满（需要增大 Q）。', g);


    start_time  = max(lastT(g), tR_m(m));
    finish_time = start_time + S_mg(m,g);

    x_rr(m, g, q) = 1;
    T_rr(g, q)    = finish_time;
    lastT(g)      = finish_time;
end

%% 
Z_rr = max(T_rr(:));

E_comp = 0;
for g = 1:G
    for q = 1:Q
        E_comp = E_comp + Wmg(:, g).' * squeeze(x_rr(:, g, q));
    end
end

E_idle = P_b * Z_rr;
eI_per_dev = eta_device * phi_device * (f_device^2) * (Nflop_device / cycle_device);
eI_true    = eI_per_dev * U;              
eU_true = sum(p_vec .* tU_vec);         
Energy_14 = eI_true + eU_true + E_idle + E_comp;
Delay     = Z_rr;
TotalSum  = Energy_14 + Delay;

%% 
fprintf('Round Robin (recomputed performance):\n');
fprintf('  Z_true (max T)              = %.6f s\n', Z_rr);
fprintf('  eI (device compute)         = %.6f J\n', eI_true);
fprintf('  eU (uplink comm)            = %.6f J\n', eU_true);
fprintf('  E_comp (server compute)     = %.6f J  (= sum Wmg*x)\n', E_comp);
fprintf('  E_idle (server idle)        = %.6f J  (= P_b * Z_true, P_b=%.4g)\n', E_idle, P_b);
fprintf('  Energy_14  = eI + eU + P_b*Z + sum(Wmg*x) = %.6f\n', Energy_14);
fprintf('  Delay      = max T                         = %.6f\n', Delay);
fprintf('  TotalSum   = Energy_14 + Delay             = %.6f\n\n', TotalSum);

%% 
gpu_of_m = nan(M,1); q_of_m = nan(M,1);
for m = 1:M
    [g_idx, q_idx] = find(squeeze(x_rr(m,:,:))>0, 1);
    if ~isempty(g_idx), gpu_of_m(m) = g_idx; q_of_m(m) = q_idx; end
end
[~, orderArr] = sort(tR_m(:), 'ascend');
arrival_rank = zeros(M,1); arrival_rank(orderArr) = (1:M)';

Ttab = table(arrival_rank, (1:M)', tR_m(:), gpu_of_m, q_of_m, ...
             'VariableNames', {'ArrivalRank','Cluster_m','Arrival_tR','GPU_g','Queue_q'});
Ttab = sortrows(Ttab, 'ArrivalRank', 'ascend');
disp(Ttab);


